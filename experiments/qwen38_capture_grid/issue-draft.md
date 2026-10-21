# [Performance discussion] When do larger CUDA graph capture sizes help?

The Qwen-3.8-Flash-Next [recipe (B300)](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next?hardware=b300) leaves CUDA graph capture sizes at their defaults (1,024-tokens). This results in prefill/mixed steps running eagerly, which is super slow due to the GDN host overhead. Adding a couple CUDA graph capture sizes improves throughput/latency significantly.

<details>
<summary>Workload and configuration details</summary>

- We used 4×B300, `Qwen/Qwen3.8-Flash-Next-FP8` and vLLM `58b3298457` (2026-10-02). We followed the recipe, with MTP3 and prefix caching enabled.
- We benchmarked with AIPerf 0.13 and SPEED-Bench throughput_1K. We allowed natural EOS, capped output at 1,024 tokens, disabled thinking and used temperature 0 and shuffle seed 42.
- We started each variant three times. We ran each concurrency three times after the first startup and once after each additional startup, giving five measurements per point. Each run submitted requests for 30 s (60 s at c256), then allowed up to 120 s for outstanding requests to finish. We reset the prefix cache before each run.
- All 90 runs passed, with 32,457 completed requests and no errors or preemptions.
- Only the capture grid changes. Up to 2K adds `1280,1536,1792,2048` to the default list. Up to 4K additionally adds `2560,3072,3584,4096`.

</details>

<p align="center">
  <img src="results/qwen38-recipe-restarts-20261020/throughput-interactivity-expanded.png" alt="Throughput versus interactivity" width="70%">
</p>

Each dot shows one run. The lines connect the averages. The orange line collapses to the "Default" performance at c256 because the mixed batches runs mostly in eager mode (see c256 below).

Across all five measurements per point, Up to 4K improves average throughput by 96.8% at c128 and 52.3% at c256 compared with Default.

<p align="center">
  <img src="results/qwen38-recipe-restarts-20261020/graph-coverage-prefill-bars.png" alt="CUDA graph coverage of prefill and mixed steps only" width="70%">
</p>

<details>
<summary>Bar chart details</summary>

* A pure decode batch can contain up to `4 × concurrency` query tokens with MTP3. Batches above that size must contain prefill work. The chart shows what percentage of these prefill/mixed batches use CUDA graphs. 
* Each bar shows the average coverage across five runs. The error bars show the minimum and maximum coverage across those runs. We include when the batch is filling up or draining. Piecewise graphs still run some operations eagerly.
* Default never uses CUDA graph at c256. A single 1,080-token prefill alongside 255 decode requests already gives us `1080 + 255 × 4 = 2100` query tokens. That is why the 2K limit is too small here.

</details>

### Trade-off

Up to 4K uses an extra 1.19 GiB/GPU for CUDA graphs.

| Variant | CUDA graph mem | KV cache mem | Token capacity | Graph capture time |
|---|---:|---:|---:|---:|
| Default | 2.85 GiB/GPU | 198.14 GiB/GPU | 26.64M tokens | 35–38 s |
| Up to 2K | 2.68 GiB/GPU (−6.0%) | 198.45 GiB/GPU (+0.2%) | 26.68M tokens (+0.2%) | 36–37 s |
| Up to 4K | 4.04 GiB/GPU (+41.8%) | 197.32 GiB/GPU (−0.4%) | 26.53M tokens (−0.4%) | 38–39 s |

The memory figures were identical at the reported precision across all three startups per variant, even after switching GPU groups. Up to 2K consistently used less CUDA graph mem and had more KV capacity than Default. This is reproducible in this setup, but we still need to investigate why. Graph capture times are shown as ranges across the three startups.

This looks like a very positive trade-off. We get a lot higher performance, while losing only little KV-capacity, and spending a few extra seconds on graph capture with Up to 4K.

## GLM 5.3 Flash

| Metric | Concurrency | Default | Up to 4K | Change |
|---|---|---:|---:|---:|
| Throughput | 64 | 3,326 tok/s | 5,147 tok/s | +54.8% |
| Throughput | 128 | 4,963 tok/s | 7,046 tok/s | +42.0% |
| TPOT | 64 | 13.89 ms | 10.72 ms | −22.8% |
| TPOT | 128 | 16.25 ms | 13.95 ms | −14.2% |

The improvements are great, but there are some drawbacks measured below (4×B300 TP4). Nevertheless the overall trade-off is very positive.

<details>
<summary>Trade-offs</summary>

| Metric | Default | Up to 4K | Change | Change (%) |
|---|---:|---:|---|---:|
| CUDA graph mem | 6.23 GiB/GPU | 12.88 GiB/GPU | +6.65 GiB/GPU | +106.7% |
| KV cache mem | 152.10 GiB/GPU | 141.41 GiB/GPU | -10.69 GiB/GPU | -7.0% |
| Token capacity | 23.63M tokens | 21.97M tokens | -1.66M tokens | -7.0% |
| Graph capture time | 38 s | 43 s | +5 s | +13.2% |
| Total startup time | 403 s | 393 s | -10 s (probably noise) | -2.5% |

</details>

## Root Cause

The cause is a discontinuity in runtime when running batches in/out of CUDA graphs. This was documented in August in the [vLLM Blog on adaptive verification](https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification) (see below), where the authors used `max_cudagraph_capture_size=4096`. The default caputure-limit of 1024 is surpassed already for 128 requests with 8 speculative tokens plus 1 bonus token (9*128=1152). And mixed prefill/decode batches fall out of the range quickly and run eagerly, even for `FULL_AND_PIECEWISE` execution. These mixed batches have very high host overhead in GDN MoE models, so they tank performance when running eagerly.

<p align="center">
  <img src="https://vllm.ai/blog-assets/figures/2026-08-14-dspark-adaptive-verification/fig2-costcurve.svg" alt="Verification and draft cost curves across CUDA graph capture sizes" width="70%">
</p>

## When larger CUDA graphs do not help

Michael Goin suggested the models that benefit from larger CUDA graph sizes are likely large complex MoE models. Our benchmarks support this reading.

### 1. Qwen3-32B BF16

Degradation: On H100 80GB, "Up to 4K" reduces throughput by 4.0% and increases TPOT by 3.4% on (macro) average across c1, c8 and c32.

<details>
<summary>Trade-offs</summary>

We used the same SPEED-Bench throughput_1K workload and AIPerf settings, with BF16 weights, TP1 and prefix caching enabled and reset before each run. Qwen3-32B has no MTP head. The values below are AIPerf's whole-run metrics, including filling and draining, averaged across three runs per variant and concurrency. We truncate the table after c32, the first concurrency to experience preemptions.

| Metric | Concurrency | Default | Up to 4K | Change |
|---|---:|---:|---:|---:|
| Throughput | 1 | 41.4 tok/s | 41.0 tok/s | −1.1% |
| Throughput | 8 | 250.7 tok/s | 258.0 tok/s | +2.9% |
| Throughput | 32 | 687.8 tok/s | 592.8 tok/s | −13.8% |
| TPOT | 1 | 23.91 ms | 24.18 ms | +1.1% |
| TPOT | 8 | 25.24 ms | 25.45 ms | +0.8% |
| TPOT | 32 | 29.88 ms | 32.33 ms | +8.2% |

The table below shows ranges across three startups per variant. Changes compare the means.

| Metric | Default | Up to 4K | Change | Change (%) |
|---|---:|---:|---:|---:|
| CUDA graph mem | 0.55 GiB/GPU | 1.18 GiB/GPU | +0.63 GiB/GPU | +114.5% |
| KV cache mem | 5.69–5.70 GiB/GPU | 4.75–4.76 GiB/GPU | −0.94 GiB/GPU | −16.5% |
| Token capacity | 46.6–46.7K tokens | 38.9–39.0K tokens | −7.7K tokens | −16.5% |
| Graph capture time | 9 s | 15–16 s | +6.3 s | +70.4% |
| Total startup time | 60–92 s | 70–102 s | +10.0 s | +14.1% |

</details>

### 2. Qwen3.5-27B BF16 (GDN)

Degradation: Failure on TP1, and same speed at TP2.
* TP1: On 1×H100 80GB with TP1 and MTP3, Default starts successfully, but "Up to 4K" fails to start. Reserving memory for the larger graph grid leaves only 238 Mamba cache blocks, fewer than the 256 sequences configured with `max_num_seqs`. vLLM rejects this configuration because there is not enough cache capacity.
* TP2: With 2×H100 80GB and TP2, both variants start successfully, but "Up to 4K" provides no clear speed-up: throughput changes ranged from −1.1% to +0.1% across c1, c8 and c32. TPOT changes ranged from −0.2% to +1.7%.

## Questions

Would it make sense to include a few larger capture sizes by default? Should this depend on the model, hardware or workload? What else should we benchmark before deciding?

## Related issues and PRs

- [#27222: Qwen3-Next CPU overhead](https://github.com/vllm-project/vllm/issues/27222) reports improved throughput with larger captures.
- [PR #49390: Raise Blackwell capture default to 1,024](https://github.com/vllm-project/vllm/pull/49390) introduced the default ceiling used here.
- [RFC #55259: Startup self-benchmarking](https://github.com/vllm-project/vllm/issues/55259) proposes measuring performance around capture-size boundaries.
