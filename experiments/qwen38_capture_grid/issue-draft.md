# [Performance discussion] Should the default CUDA graph grid include larger sizes?

The [B300 deployment recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next?hardware=b300) leaves CUDA graph capture sizes at their defaults (1,024-tokens). This results in prefill/mixed steps running eagerly, which is super slow due to the GDN host overhead. Adding a couple CUDA graph capture sizes improves throughput/latency significantly. Probably this should be the default (at least for GDN).

<details>
<summary>Workload and configuration details</summary>

- We used 4×B300, `Qwen/Qwen3.8-Flash-Next-FP8` and vLLM `58b3298457` (2026-10-02). We followed the recipe, with MTP3 and prefix caching enabled.
- We benchmarked with AIPerf 0.13 and SPEED-Bench throughput_1K. We allowed natural EOS, capped output at 1,024 tokens, disabled thinking and used temperature 0 and shuffle seed 42.
- We started each variant three times. We ran each concurrency three times after the first startup and once after each additional startup, giving five measurements per point. Each run submitted requests for 30 s (60 s at c256), then allowed up to 120 s for outstanding requests to finish. We reset the prefix cache before each run.
- All 90 runs passed, with 32,457 completed requests and no errors or preemptions.
- Only the capture grid changes. **Up to 2K** adds `1280,1536,1792,2048` to the default list. **Up to 4K** additionally adds `2560,3072,3584,4096`.

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
* We combine the logged step counts from all five runs per point, including when the batch is filling up or draining. Piecewise graphs still run some operations eagerly.
* Default never uses CUDA graph at c256. A single 1,080-token prefill alongside 255 decode requests already gives us `1080 + 255 × 4 = 2100` query tokens. That is why the 2K limit is too small here.

</details>

### Trade-off

Up to 4K uses an extra **1.19 GiB/GPU** for CUDA graphs.

| Variant | CUDA graph mem | KV cache mem | Token capacity | Graph capture time |
|---|---:|---:|---:|---:|
| Default | 2.85 GiB/GPU | 198.14 GiB/GPU | 26.64M tokens | 35–38 s |
| Up to 2K | 2.68 GiB/GPU (−6.0%) | 198.45 GiB/GPU (+0.2%) | 26.68M tokens (+0.2%) | 36–37 s |
| Up to 4K | 4.04 GiB/GPU (+41.8%) | 197.32 GiB/GPU (−0.4%) | 26.53M tokens (−0.4%) | 38–39 s |

The memory figures were identical at the reported precision across all three startups per variant, even after switching GPU groups. Up to 2K consistently used less CUDA graph mem and had more KV capacity than Default. This is reproducible in this setup, but we still need to investigate why. Graph capture times are shown as ranges across the three startups.

This looks like a very positive trade-off. We get a lot higher performance, while losing only little KV-capacity, and spending a few extra seconds on graph capture with Up to 4K.

### GLM 5.3 Flash

| Metric | Concurrency | Default | Up to 4K | Change |
|---|---|---:|---:|---:|
| Throughput | 64 | 3,326 tok/s | 5,147 tok/s | +54.8% |
| Throughput | 128 | 4,963 tok/s | 7,046 tok/s | +42.0% |
| TPOT | 64 | 13.89 ms | 10.72 ms | −22.8% |
| TPOT | 128 | 16.25 ms | 13.95 ms | −14.2% |

The improvements are great, but there are some drawbacks measured below (4×B300 TP4). Nevertheless the overall trade-off is very positive.

| Metric | Default | Up to 4K | Change | Change (%) |
|---|---:|---:|---|---:|
| CUDA graph mem | 6.23 GiB/GPU | 12.88 GiB/GPU | +6.65 GiB/GPU | +106.7% |
| KV cache mem | 152.10 GiB/GPU | 141.41 GiB/GPU | -10.69 GiB/GPU | -7.0% |
| Token capacity | 23.63M tokens | 21.97M tokens | -1.66M tokens | -7.0% |
| Graph capture time | 38 s | 43 s | +5 s | +13.2% |
| Total startup time | 403 s | 393 s | -10 s (probably noise) | -2.5% |

### Questions

Would it make sense to include a few larger capture sizes by default? Should this depend on the model, hardware or workload? What else should we benchmark before deciding?

### Related issues and PRs

- [#27222: Qwen3-Next CPU overhead](https://github.com/vllm-project/vllm/issues/27222) reports improved throughput with larger captures.
- [PR #49390: Raise Blackwell capture default to 1,024](https://github.com/vllm-project/vllm/pull/49390) introduced the default ceiling used here.
- [RFC #55259: Startup self-benchmarking](https://github.com/vllm-project/vllm/issues/55259) proposes measuring performance around capture-size boundaries.
