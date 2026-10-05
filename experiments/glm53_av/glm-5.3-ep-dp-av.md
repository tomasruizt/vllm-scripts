# GLM-5.3: DEP4 and adaptive verification

Tested October 2, 2026, on unmodified [vLLM main at 58b329845][commit].

vLLM supports DP + EP + adaptive verification (AV). We tested full GLM-5.3 NVFP4 on four B300s with no speculation, DSpark, and DSpark + AV. DFlash2 is another available speculator, but we have not tested it locally. The research did not find a published AV-on/off comparison for full GLM-5.3.

## Results

**Throughput (output tokens/sec).** Bsz is total client concurrency across all four DP ranks.

| Bsz | No speculation | DSpark, AV off | DSpark + AV |
| ---: | ---: | ---: | ---: |
| 8 | 430.3 | 1,121.2 | 1,116.1 |
| 64 | 2,082.3 | 5,074.7 | 4,536.0 |
| 128 | 2,087.4 | 7,682.0 | 7,764.5 |
| 256 | 2,742.4 | 7,836.0 | 11,279.5 |

**Accuracy and acceptance length.**

| Method | Bsz | GSM8K accuracy | AL |
| --- | ---: | ---: | ---: |
| No speculation | 8 | 91.58% | N/A |
| No speculation | 64 | 91.36% | N/A |
| No speculation | 128 | 91.05% | N/A |
| No speculation | 256 | 90.60% | N/A |
| DSpark, AV off | 8 | 90.83% | 4.894 |
| DSpark, AV off | 64 | 91.05% | 4.859 |
| DSpark, AV off | 128 | 92.19% | 4.817 |
| DSpark, AV off | 256 | 91.13% | 4.930 |
| DSpark + AV | 8 | 90.67% | 4.691 |
| DSpark + AV | 64 | 90.83% | 4.633 |
| DSpark + AV | 128 | 91.21% | 4.276 |
| DSpark + AV | 256 | 91.43% | 4.290 |

All methods used the same target and serving settings. Both DSpark methods used the same draft checkpoint and eight draft tokens; only the AV flag changed.

Each run evaluated all 1,319 GSM8K questions with five-shot completion prompts, temperature 0, seed 42, and a 2,048-token output limit. There were no API errors. Two runs each had one unparseable answer: no speculation at bsz 64 and AV off at bsz 128.

These are single runs after warmup. Each method reused one server for 8→64 and another for 128→256. Prefix caching stayed enabled, so the second run reused the same prompts. A few slow final requests affected throughput, especially with no speculation, AV off at bsz 256, and AV on at bsz 64. We have not isolated the cause or established a quality difference.

AL is `1 + accepted_tokens / draft_steps`, summed across ranks with warmup excluded. It measures accepted progress, not verification cost. Draft counters count proposals before AV trims them, so they cannot tell us how much verification AV saved.

## DEP4 and AV

DEP4 means TP1 / DP4 / EP4: each GPU handles its own requests, attention weights are replicated, and routed experts are shared across four GPUs. EP size is `TP × DP`; it does not add another GPU multiplier. TP4 / EP4 uses the same four GPUs but runs one tensor-parallel batch. See the [DP guide][dp-doc] and [EP guide][ep-doc].

AV decides how many proposed tokens are worth verifying on each step. It uses execution costs measured at startup and estimates of draft acceptance. DSpark uses its trained confidence head; MTP, EAGLE3, DFlash, and speculators without a confidence head can use the online estimator added in [PR #52228][estimator-pr]. Both feed the same [AV allocator][av-code]. The docs at the tested commit still say “DSpark only,” although the code supports other methods.

Each DP rank chooses its own budget, then the ranks agree on a graph shape large enough for all of them. Padding can reduce the savings when their budgets differ. We did not measure that cost separately.

## Models and configs

We used [RedHatAI/GLM-5.3-NVFP4][target] and [RedHatAI/GLM-5.3-speculator.dspark][draft]. [Inco's GLM-5.3-DFlash2][dflash] is the non-DSpark alternative; native MTP is also available. These are for full GLM-5.3, not GLM-5.3-Flash.

Full GLM-5.3 FP8 does not fit on four H200s. The [official recipe][glm-recipe] uses eight; NVFP4 made our four-B300 setup practical. GLM's Blackwell sparse-attention backend supports AV graph shapes. The Hopper backends inspected at this commit did not, so DSv4 AV support on H200 does not establish GLM AV support there.

AV needs Model Runner V2, a compatible attention backend, and full decode CUDA graphs. Eager-only mode, LoRA, and pipeline parallelism are unsupported here. Non-DSpark AV also needs draft logits and conflicts with `use_local_argmax_reduction`. See the [configuration checks][config].

Use a GPU reservation to set `CUDA_VISIBLE_DEVICES`, then launch with:

```bash
serve_glm53() {
  VLLM_USE_V2_MODEL_RUNNER=1 vllm serve RedHatAI/GLM-5.3-NVFP4 \
    --revision c8917e4258572c405575855ff53effe58c17a38e \
    --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
    --all2all-backend allgather_reducescatter \
    --attention-backend FLASHINFER_MLA_SPARSE \
    --kv-cache-dtype fp8_e4m3 --block-size 64 \
    --max-model-len 16384 --max-num-seqs 128 --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.85 \
    --reasoning-parser glm45 --chat-template-content-format string \
    --trust-remote-code --disable-uvicorn-access-log \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","max_cudagraph_capture_size":1152}' \
    --speculative-config "$1"
}
```

DSpark + AV, tested locally:

```bash
serve_glm53 '{"method":"dspark","model":"RedHatAI/GLM-5.3-speculator.dspark","revision":"b374b95663447ea0e935151be4f3d6666e36e6d7","num_speculative_tokens":8,"attention_backend":"FLASH_ATTN","draft_sample_method":"probabilistic","enable_adaptive_verification":true}'
```

DFlash2 + AV, not tested locally:

```bash
serve_glm53 '{"method":"dflash","model":"incoai/GLM-5.3-DFlash2","num_speculative_tokens":7,"attention_backend":"FLASH_ATTN","draft_sample_method":"probabilistic","enable_adaptive_verification":true}'
```

For DSpark without AV, change only the flag to `false`. For no speculation, omit `--speculative-config`.

## Related experiments

- [Lucas's DSv4-Flash experiment, PR #52795][dsv4-pr]: DSpark with seven drafts, TP2 / DP2 / EP4 on four H200s. It compared AV against fixed verification; its separate GSM8K run used TP4 / EP4.
- [GLM-5.2 experiment, PR #52783][glm52-pr]: DSpark on four B300s, TP4 / DP1 / EP4. AV helped at every tested concurrency, with larger gains at higher load. This was not DEP4.
- [PR #52228][estimator-pr]: AV results with MTP, DFlash, and DFlash2, but no GLM-5.3 result.

The [GLM-5.3 optimization blog][glm-blog] mentions AV as planned follow-up work. The released DSpark model card has speculation benchmarks, but its example leaves AV off.

## Questions

**1. Must AV be explicitly enabled for DSpark?** Yes. `enable_adaptive_verification` defaults to `false`; a confidence head does not enable it automatically. [Config][spec-config].

**2. What changes within each DP rank?** The proposal width stays at eight. Each step, AV chooses a total verification budget and divides it among requests: one can get zero tokens, another two, another eight. Startup profiles costs, not a fixed budget. CPU selection uses slightly stale confidences; GPU allocation uses current ones. [Allocator][av-code].

**3. Doesn't DP balance requests? Why padding?** Balanced arrivals do not mean identical batches: requests differ in length, completion time, and acceptance estimates. The largest query count sets the shared graph shape, so smaller ranks may do padded work. [Coordination][dp-code].

**4. What does `dispatch_cg_and_sync_dp()` synchronize?** The next forward's execution mode and graph shape. Expert communication needs matching collective sequences, including idle ranks. Graph execution pads token rows and some request metadata, not every request's historical KV context. EP can handle different real batch sizes in eager mode. [Code][dp-code].

**5. Are DFlash and DSpark K=1 drafters?** They propose multiple tokens in one backbone pass. K means draft tokens, not passes. DSpark also uses lightweight sequential Markov-head sampling. AV can reduce target verification even when drafting takes one pass. [DSpark code][dspark-code].

**6. Is expert synchronization why batch-size dynamic-K is disabled with DP?** It is part of the reason, but consistent draft/target control flow also matters. The guard disables `num_speculative_tokens_per_batch_size` for all DP>1 setups. AV keeps the configured draft structure and changes verification work. Our DSpark's draft backbone is dense, so draft-expert synchronization alone does not explain the restriction. [Guard][config].

**7. Where are the experimental sources?** The three PRs above contain the public results. Our GLM-5.3 measurements are local; saved inputs are listed below.

**8. Did Lucas's DSv4 experiment use speculation?** Yes. Both methods used DSpark with seven tokens and probabilistic sampling; only verification changed. [PR #52795][dsv4-pr].

**9. Do the GLM-5.2 results prove speculation stays active at high load?** DSpark stayed configured, but AV can verify zero drafts on some steps while the drafter still runs. Throughput cannot show how often. Our counters prove some drafts were accepted, not that every step used them. [Results][glm52-pr].

**10. Can we monitor tokens actually verified?** There is no dedicated counter here. Draft counters count proposals before AV trims them. `--cudagraph-metrics` logs real and padded query counts after AV selection, but includes prefill and non-draft tokens. We did not enable it. For precise savings, record the admitted draft budget and padded query count per rank and step. Total budgets are CPU-visible; per-request lengths are on the GPU. [Graph metrics][graph-metrics] and [allocator][av-code].

**11. Should disabling AV hurt throughput at high bsz?** It can, when verifying the whole block costs more than its accepted tokens are worth. High acceptance, AV overhead, and DP padding can change the tradeoff. Repeated runs with budget and padding measurements would help explain it.

## Files

`experiments/glm53_av/report/index.html` in `vllm-scripts` contains the throughput/interactivity chart and the two tables in tabs. It uses the [original B200 theme](https://tomasruizt.github.io/reports/b200-dflash/). Interactivity uses mean request TPOT from saved Prometheus sums and counts; the B200 report uses p90.

Code: `experiments/glm53_av/analysis/`. Shared plotting and styles: `reporting/`. From the `vllm-scripts` checkout, rebuild with:

```bash
~/.venv/bin/python experiments/glm53_av/analysis/build_report.py
```

`experiments/glm53_av/report/data/` preserves results, metric snapshots, client logs, run scripts, and launch commands. Original server logs are under `/tmp/glm53-dep4-{baseline,dspark-avoff,av}-20261002` and `/tmp/glm53-dep4-c8-c64-20261002`. Generated reports and input data stay local and are ignored by Git. The AV observability review is alongside these notes in `observability-AV-pr-reviev.md`.

Model cache: `/home/tomasruizt/code/vllm/.cache/huggingface/hub` (436 GiB). The shared `/data/engine/hub_cache` had full GLM-5.3 FP8, but not the matching NVFP4 target or DSpark checkpoint. All benchmark servers stopped and GPU reservations were released.

[commit]: https://github.com/vllm-project/vllm/commit/58b3298457dde7b4554b3b4e20b238c0ac2c3a65
[target]: https://huggingface.co/RedHatAI/GLM-5.3-NVFP4/tree/c8917e4258572c405575855ff53effe58c17a38e
[draft]: https://huggingface.co/RedHatAI/GLM-5.3-speculator.dspark/tree/b374b95663447ea0e935151be4f3d6666e36e6d7
[dflash]: https://huggingface.co/incoai/GLM-5.3-DFlash2
[dp-doc]: https://github.com/vllm-project/vllm/blob/58b3298457dde7b4554b3b4e20b238c0ac2c3a65/docs/serving/data_parallel_deployment.md
[ep-doc]: https://github.com/vllm-project/vllm/blob/58b3298457dde7b4554b3b4e20b238c0ac2c3a65/docs/serving/expert_parallel_deployment.md
[av-code]: https://github.com/vllm-project/vllm/blob/58b3298457dde7b4554b3b4e20b238c0ac2c3a65/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py
[spec-config]: https://github.com/vllm-project/vllm/blob/58b3298457dde7b4554b3b4e20b238c0ac2c3a65/vllm/config/speculative.py
[config]: https://github.com/vllm-project/vllm/blob/58b3298457dde7b4554b3b4e20b238c0ac2c3a65/vllm/config/vllm.py
[dp-code]: https://github.com/vllm-project/vllm/blob/58b3298457dde7b4554b3b4e20b238c0ac2c3a65/vllm/v1/worker/gpu/dp_utils.py
[dspark-code]: https://github.com/vllm-project/vllm/blob/58b3298457dde7b4554b3b4e20b238c0ac2c3a65/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py
[graph-metrics]: https://github.com/vllm-project/vllm/blob/58b3298457dde7b4554b3b4e20b238c0ac2c3a65/vllm/compilation/cuda_graph.py
[estimator-pr]: https://github.com/vllm-project/vllm/pull/52228
[dsv4-pr]: https://github.com/vllm-project/vllm/pull/52795
[glm52-pr]: https://github.com/vllm-project/vllm/pull/52783
[glm-recipe]: https://recipes.vllm.ai/zai-org/GLM-5.3
[glm-blog]: https://vllm-project.github.io/2026/09/08/glm53-part1-hybrid-sparse-offloading.html
