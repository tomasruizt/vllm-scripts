# Qwen3.5 DFlash adaptive verification (PR #60229): findings

Shared record for all sessions working on this experiment. Read it first, append new results with a date, and mark superseded claims instead of deleting them.

Setup: Qwen3.5-35B-A3B BF16, DFlash k=7, one B300 per server (TP=1), SPEED-Bench throughput-1K, AIPerf 0.13.0, PR commit `888ec6bb`. Human-facing summary: `av-cost-model.html` (published as the "Why AV Slows Down c128" artifact).

## Established (as of 2026-10-18)

### Benchmark method

- The original 5 s duration + 10 s grace cells cancelled most requests at c256, so they only counted short requests. Replaced by fixed request counts, steady-state window metrics and validity gates (`run-steady.sh`, `steady-speedbench.yaml`, `build_report.py`).
- c256 does not fit: the hybrid KV cache holds ~150 sequences at `--gpu-memory-utilization 0.92`. Sweeps stop at c128.

### Throughput (medians of 3 repeats, tok/s)

| c | AV off | AV on | Δ | AL off / on |
|---|---|---|---|---|
| 1 | 482 | 482 | +0.1% | 2.71 / 2.61 |
| 8 | 1,958 | 1,986 | +1.4% | 2.96 / 2.78 |
| 32 | 3,998 | 3,951 | −1.2% | 3.02 / 2.69 |
| 64 | 5,513 | 5,346 | −3.0% | 2.89 / 2.55 |
| 128 | 7,504 | 7,006 | −6.6% | 2.94 / 2.34 |

### Step structure

- Two step populations: decode-only (~20 to 25 ms at c128) and steps containing a prefill (~78 to 90 ms). Proof: 99 to 100% of slow steps are followed by a request's first token, against 0.1% of fast steps.
- Prefill step time does not grow with the number of prompts in the step (1, 2 or 3 prompts of ~1,080 tokens all take ~85 to 90 ms at c128). It is fixed overhead, not compute (1K tokens ≈ 3 ms at B300 peak). Prefill steps are ~33% of steps but ~68% of time at c128.
- Steps with a prefill never use full CUDA graphs (piecewise at ≤1,024 tokens, no graphs above). With a 16K token budget prompts are prefilled whole, not chunked.
- 2026-10-18 (prefill-profiling session, torch profiler, `profile-prefill-step.sh`, traces in `results/prefill-step-profile/`): the ~85 ms prefill step is almost all CPU overhead, the GPU is busy ~19 ms. Every prefill step here exceeds `max_cudagraph_capture_size=1024` (B300 default), so it runs fully eager, not even piecewise. ~2.4 ms CPU per layer × 40 layers: flashinfer `trtllm_bf16_moe` Python wrapper gaps (~0.8 ms/layer), GDN eager ops plus CuTe-DSL launch cost (~1.2 ms/layer), compiled pieces without graphs. py-spy attribution: done, see below. Proposed fix: add a few large capture sizes (1,536 to 6,144) so prefill steps get piecewise graphs, then rerun AV off/on to see whether the c128 gap shrinks.
  - Torch profiler, AV off, per step (profiler inflates CPU, 109 ms here vs ~85 to 90 ms from benchmark ITL). GPU time scales with prompt tokens, CPU time does not:

    | step | period ms | CPU `execute_model` ms | GPU busy ms | kernels |
    |---|---|---|---|---|
    | 1 prompt (1,080 tok) | 109 to 113 | 105 to 109 | 19 | 1,529 |
    | 3 prompts (3,240 tok) | 103 | 95 | 30 | 1,437 |
    | 5 prompts + 3 decodes (5,424 tok) | 144 | 132 | 48 | 1,966 |
    | decode-only, 8 req (64 tok, full graph) | 8.5 | 5 | 5 to 9 | ~1,100 |

  - The whole 109 ms is the target forward (layers from 12 to 108 ms). ~70 µs CPU per kernel launch, vs ~5 to 10 µs for plain eager PyTorch.
  - Why 1,024: not set by us. vLLM default is `min(max_num_seqs × (1+k) × 2, 1024 on Blackwell / 512 elsewhere)` (`vllm/config/vllm.py`, raised to 1,024 in PR #49390, which measured 80 s and 4.88 GiB to capture up to 1,024 on DSV4-Flash). With spec decode the uniform-decode sizes are appended above 1,024, so decode-only steps get full graphs, but the general piecewise grid stops at 1,024.
  - py-spy (`make profile-prefill PROF_MODE=pyspy`, `analyze_pyspy.py`, `results/prefill-step-profile/pyspy-prefill.txt`): 348 samples of the engine main thread inside `execute_model` during back-to-back solo 1,080-token prefills. No single hotspot, the overhead is spread over Python-heavy launch paths:
    - Compiled pieces running without a graph (`cuda_graph.py` wrapper falls through): 55%. These are what piecewise graphs would replay.
    - MoE (`_moe_forward_shared`, inside the compiled pieces): 39%. flashinfer `trtllm_bf16_moe` C++ launch ~7%, autotuner lookup per call ~3%, shared experts ~16%.
    - GDN core (splitting op, always eager): 38%. CuTe-DSL `chunk_gated_delta_rule` ~10%, Triton `causal_conv1d` ~5%, Triton gated layernorm ~5%, vLLM indexing glue ~12%.
    - Full attention: 4%.
    - By package of the deepest Python frame: flashinfer 26%, Triton launch path 13%, vLLM GDN glue 12%, rest torch, Inductor, Dynamo. flashinfer's `_auto_dump_wrapper` is on 27% of stacks but has ~0 self time (it is just the API decorator).
    - Caveats: py-spy took ~1 s per sample on this process (only 387 samples, ±2.5 points per bucket). `--native` fell 380 s behind and is unusable. Without it, time in C extensions lands on the calling Python line (e.g. 6% at vLLM `Parameter.__torch_function__` is really the cuBLAS launch of `F.linear`).
    - Unverified: `initial_state[~prefill_has_initial_state, ...] = 0` (`qwen_gdn_linear_attn.py:1510`) forces a host sync per GDN layer if the mask lives on GPU. Cheap here since the GPU is idle, but it stops the CPU from running ahead.
  - Expected effect of larger capture sizes (estimate, not measured): removes the ~55% in compiled pieces, so a prefill step drops from ~90 to roughly 40 ms. The GDN ~38% remains and needs either a graph-safe GDN prefill (full graphs) or less Python per call.
  - Model size matters: with ~3B active params on a B300 a 1K prefill is CPU-bound. With a larger model prefill becomes GPU-bound, the grouping penalty shrinks and pruning saves more per token. GLM-5.3 (`../glm53_av`, DEP4, DSpark) shows AV −11% at bsz 64, ~flat at 128, +44% at 256, but that comparison is not controlled.

### AV decisions (per-step trace, c128)

- Decode-only steps: AV admits 28% of drafts (252 of 891). Prefill steps: AV admits 84% (98% at c32). Forcing 100% in prefill steps (`VLLM_AV_FULL_VERIFY_ON_PREFILL=1`) gives no gain: c128 7,006 → 6,945 tok/s, AL 2.34 → 2.40.
- Budget and output trace records have no shared step id. Matching them by a fixed offset is unreliable (produced > verified in 46% of matched steps), so per-step-type token counts are unknown until the trace tags both with a step id.
- 2026-10-19, AV overhead: `k=2` control at c128 (`qwen-av-k2-control-20261018`, output records only): AV off with 128 × 3 tokens takes 20.7 ms median, AV on (pruned, ~2.5 tokens per request) 21.1 ms, AV off `k=7` 28.0 ms. AV on produces 321 tokens per step vs 285 for `k=2`. nsys (3 s window at c128, `qwen-av-nsys-20261018`): AV-only kernels take ~0.15 ms per step (`_local_max_sumexp_kernel` ~116 µs, top-k/sort/cumprod/masking ~30 µs). GPU busy ~68% of the window in both modes.
- 2026-10-19, requests vs tokens: fit over ~10K AV-off decode steps (k=7 and k=2, c8 to c128): step ≈ 8.5 ms + 50 µs × requests + 13 µs × tokens (R² 0.70, rough). At fixed 128 requests, 3 → 8 tokens per request adds 7.1 ms.

### Cost model

- Startup profile (no debug logging): verify cost rises to 19.4 ms at 512 tokens, then falls to ~11.8 ms at 1,024. The dummy batch uses `min(N, 512)` requests, so from 512 to 1,024 requests switch from 1 to 2 query tokens, and 1-token requests are profiled as more expensive. The non-decreasing clamp hides the dip as a flat 19.4 ms.
- Decode-only steps: level too low (336 tokens/step: predicted 16.4 ms, measured 20.8 ms) and marginal cost too high (336 → 1,024 tokens/step: predicted +5.5 ms, measured +4.2 ms). Both bias AV toward over-pruning.
- Prefill steps: predicted ~40 ms, measured ~85 to 90 ms, nearly flat in tokens. AV still admits most drafts there.
- 2026-10-18, traced c8 to c128 (`qwen-av-diag-curve-20261018`, `plots/decode-steps.csv`): at equal tokens/step, AV-on decode steps are slower than AV-off ones (15.2 vs 12.1 ms at ~150 tokens, 22.2 vs 18.9 ms at ~340 to 380). AV-on spreads tokens over 3× more requests (2.6 vs 8 tokens each). Step time is driven mainly by request count, not tokens, so pruning drafts saves little (c128: 1,024 tokens 28.0 ms vs AV's ~336 tokens 22.2 ms).

### Why AV is slower at c128 (partial)

- AV's own per-step overhead is small (see AV decisions). Whether its shorter decode steps make up for fewer tokens is unknown (needs step-id tracing).
- Hypothesis, weak evidence: smaller AV decode steps finish fewer requests per step, so fewer prompts share each fixed-cost prefill step (prompts per prefill step 1.40 → 1.32, prefill steps +5.2% at c128, client-side reconstruction). Toy simulations were removed: they did not demonstrate the effect convincingly.
- Main open question: how much of the gap survives when prefill steps run piecewise instead of eager (capture sizes above 1,024).
- **Answered 2026-10-19** (`qwen-speedbench-cg-20261018`, run by the prefill-profiling session, 3 repeats, all 48 cells valid, summary in `report/cg/`): with 14 extra capture sizes from 1,280 to 6,144 (`*cg` modes), AV on vs off is within noise at every c. c128: 9,364 vs 9,349 tok/s (−0.2%), c64 +1.5%, c32 −1.2%. Without the extra sizes in the same sweep: c128 −3.7%, c64 −4.0%. The extra graphs give +26 to 37% throughput at c32 to c128 in both modes. AL is unchanged (c128: 2.94 off, 2.41 on). Conclusion: the c128 AV slowdown is an artifact of eager, CPU-bound prefill steps above the 1,024-token capture limit.

## Runner pitfalls

- 2026-10-19: `run-steady.sh` used fixed ports 8630/8631 for every run. Two concurrent runs collided: the second run's servers failed to bind, its health check passed against the first run's servers, and its clients loaded them. Only `qwen-av-steps-20261019` and the first attempt of `qwen-speedbench-steady-cg-20261019` were affected (both discarded and rerun). All earlier runs were checked: no overlap and no bind conflicts in their server logs. Fix: `PORT_A`/`PORT_B` env vars, and `start_server` aborts if the port already serves something. Use distinct ports for concurrent runs.
- Killing a run: kill the main `run-steady.sh` process tree first, otherwise the launch subshells respawn workers. Then `canhazgpu release --gpu-ids ...` for stale reservations.

## Superseded claims

- "Debug cost-table dumps show a flat 17 ms up to 1,024 tokens": artifact of `VLLM_LOGGING_LEVEL=DEBUG` slowing small profiling runs. Superseded by the trace profile.
- "Full verify on prefill does not help" from the first shortcut sweep (`qwen-speedbench-pf-20261016`): that sweep ran unpatched code because servers started inside the vLLM checkout. Rerun as `qwen-speedbench-pf2-20261016` with the same conclusion.
- "AV prunes too much in mixed steps": wrong, it admits 84 to 98% there.
- "Grouping explains the whole gap": overstated, see above.

## Latent issues

- AV profiling dummy runs pass `is_padding=True` for all tokens (`model_runner.py`, `is_padding=not is_profile` with `is_profile=False`), and `VLLM_MOE_SKIP_PADDING` defaults to 1. MoE backends that use vLLM's top-k router would skip expert compute while profiling, making the profile too cheap. This run used `TrtLlmBf16ExpertsMonolithic`, which routes inside its kernel and likely ignores the mask (not verified line by line).

## Open TODOs

1. Torch-profile 512 dummy requests with 1 vs 2 query tokens to find why `q = 1` is slower.
2. Check whether routing `q = 1` requests through the `q ≥ 2` path speeds up plain decode.
3. Done 2026-10-18: profile one step with a single ~1K prefill (see Step structure). Next: rerun the c128 AV off/on sweep with `-cc.max_cudagraph_capture_size=8192` or a few explicit large sizes (1,536 to 6,144), and measure prefill step time and the AV gap.
4. Check whether full CUDA graphs can be enabled for mixed prefill + decode batches.
5. Paired, traced AV off and on runs on the same GPU (3 repeats) to split the c128 gap.
6. Done 2026-10-18: denser decode-only curve (see Cost model). Model now at `/data/tomasruizt/hub_cache` (shared cache copy was removed), AIPerf 0.13.0 venv at `~/aiperf-venv` (the `/tmp` one was cleaned up).

## Runs (remote `~/runs/`)

- `qwen-speedbench-steady-20261016`: main sweep, AV off and on, c1 to c128, 3 repeats.
- `qwen-speedbench-pf2-20261016`: full verify on prefill, 3 repeats, plus AV on control.
- `qwen-av-diag-20261016`: traced AV on and off, c32 and c128.
- `qwen-av-diag-curve-20261018`: traced AV on and off, c8 to c128 (7 levels).
- `qwen-speedbench-cg-20261018` (launched 2026-10-18, queued for GPUs 1 and 2): AV off/on × default capture sizes vs `*cg` (default list plus 1,280 to 6,144, see `run-steady.sh`), c1/c32/c64/c128, 3 repeats. off and offcg (and on and oncg) run back to back on the same GPU per repeat.
- `qwen-prefill-step-profile`: AV off, torch profiler traces (`solo`, `mixed`) and py-spy stacks of solo 1K prefills. The "mixed" scenario did not overlap with decodes as designed (background requests finished first).

## Upstream

- #54046 and PR #54065: AV tail profile shapes the scheduler cannot produce, proposed fix adds prefill chunks.
- #52057: startup profile drift and context-length error.
- #54749: RFC, speculation length keyed on batch size only.
- #51874, PR #58340, PR #60430: metrics for verified draft counts.
- PR #47808 (merged): introduced the cost model. PR #48692 (closed): fixed-K fallback on prefill batches.

## 2026-10-19: Qwen3.8-Flash-Next capture-grid analysis (no new measurements)

- Model configuration pinned to `de4b8e4d43b917e7706784d8bb445c9af86a3540`: 48 layers (36 GDN, 12 QSA), 6B active main-model parameters, four residual streams, 51B n-gram embeddings and MTP. The BF16 checkpoint is about 335 GiB, so the previous single-B300 BF16 setup does not transfer. Sources: https://huggingface.co/Qwen/Qwen3.8-Flash-Next and https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next.
- The NVIDIA source enables breakable CUDA graphs for `Qwen4ExpForConditionalGeneration`. QSA, PLE convolution and CPU-offloaded embedding prefetch have explicit eager breaks. Larger captures can bypass surrounding launch overhead, but do not capture every operation or remove offload cost (`vllm/config/vllm.py`, `models/qwen4_exp/nvidia/qsa.py`, `nvidia/ple_layer.py`, `common/ngram_embedding.py`).
- For the recipe's max_num_seqs=256 and MTP3, target verification has at most 1,024 tokens, already within the Blackwell default capture cap. Extension primarily targets prefill and mixed steps. The QSA 2,048-token retrieval budget and 262K context limit are separate from scheduled query-token capture sizes.
- Candidate sparse additions to the resolved default list: `[1280,1536,1792,2048,2560,3072,3584,4096]`. For the inspected Blackwell MTP3 configuration this is 83 to 91 token-size entries, versus 275 for a dense stride-16 grid through 4,096. These are list entries, not total runtime graph counts. Explicit lists replace defaults, so preserve the original sizes.
- Recommendation: compare default, sparse-to-2,048 and sparse-to-4,096 with fixed topology, precision, offload, MTP and scheduler budget. Measure mixed-step graph coverage, eager CPU/GPU time, padding, throughput/TTFT/TPOT, startup, graph memory and cache capacity. Extend beyond 4,096 only if material eager time remains above it. The earlier 25 to 37% gain and graph-memory costs are Qwen3.5 measurements, not predictions for this model. No global default change is justified yet.

## 2026-10-19: Qwen3.8-Flash-Next capture-grid measurements

Three fresh servers on the same four B300s, one server per variant, clean vLLM `58b3298457dde7b4554b3b4e20b238c0ac2c3a65`, BF16 checkpoint revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`. Native AIPerf 0.13.0 SPEED-Bench throughput-1K, c32/c128, seed 42, streaming, temperature 0, thinking disabled, natural EOS capped at 1,024 tokens. Each cell submits for 30 s with 120 s drain grace. All six cells passed completion, saturation, queueing, preemption and window-length gates: 1,933 requests, zero errors/cancellations/preemptions. No accuracy evaluation. GPU reservation lasted 15m58s, all four workers exited and GPUs were released.

| Variant | c | Requests | Steady tok/s | Whole-run AIPerf tok/s | Steady TPOT ms | Steady TTFT ms |
|---|---:|---:|---:|---:|---:|---:|
| Default | 32 | 160 | 2,984 | 2,954 | 8.88 | 165.6 |
| Up to 2K | 32 | 223 | 4,376 | 4,153 | 6.73 | 84.7 |
| Up to 4K | 32 | 227 | 4,428 | 4,184 | 6.73 | 72.1 |
| Default | 128 | 362 | 4,816 | 5,912 | 16.44 | 252.0 |
| Up to 2K | 128 | 468 | 8,000 | 7,818 | 12.84 | 153.3 |
| Up to 4K | 128 | 493 | 9,231 | 8,177 | 11.72 | 124.4 |

| Variant | Size entries | Actual graph GiB/GPU | KV tokens | Final capture s | Total startup s |
|---|---:|---:|---:|---:|---:|
| Default | 83 | 2.28 | 9,616,822 | 36 | 305 |
| Up to 2K | 87 | 2.41 | 9,602,779 | 35 | 196 |
| Up to 4K | 91 | 4.03 | 9,537,828 | 40 | 183 |

Up to 2K improves steady throughput by 46.6%/66.1% at c32/c128, for +0.13 GiB/GPU and -0.15% KV capacity. Up to 4K adds 1.2%/15.4% over Up to 2K, for another 1.62 GiB/GPU (-0.82% KV capacity versus Default). Whole-run AIPerf gains versus Default are 40.6%/32.2% for Up to 2K and 41.6%/38.3% for Up to 4K. Recommendation: increase coverage for this deployment, Up to 2K captures almost all c32 benefit and Up to 4K is the best tested c128 configuration. This is one pass per cell, not evidence for a universal default or a precise incremental 4K gain. Steady windows are only 10.6 to 23.3 s. The first startup includes cold JIT work, so total startup differences cannot be attributed to the grid.

Built-in dispatch statistics over both cells per server: Default FULL/NONE/PIECEWISE = 1919/321/1, Up to 2K = 2491/105/352, Up to 4K = 2538/28/451. Every NONE step exceeded its variant's cap. Padding overhead by scheduled tokens rose from 0.26% to 3.76% to 5.82%. This verifies increased graph coverage, without a new CPU/GPU profiler attribution.

Portable serving equivalent, with all measured flags (the actual launcher used the pinned local snapshot and `python -m vllm.entrypoints.openai.api_server`):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_USE_V2_MODEL_RUNNER=1 vllm serve Qwen/Qwen3.8-Flash-Next \
  --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --served-model-name qwen38 --language-model-only --dtype bfloat16 \
  --tensor-parallel-size 4 --enable-expert-parallel --gpu-memory-utilization 0.90 \
  --max-model-len 16384 --max-num-seqs 256 --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching --disable-uvicorn-access-log --cudagraph-metrics \
  --kv-cache-dtype fp8 --attention-config.indexer_kv_dtype fp8 \
  --engram-config '{"cpu_offload":true}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --host 127.0.0.1 --port 8748
```

Default adds no compilation flag. For Up to 2K or Up to 4K, append `--compilation-config "{\"cudagraph_capture_sizes\":[$SIZES]}"`, with the complete list generated as follows:

```bash
SIZES=$(echo 1 2 4 $(seq 8 8 248) $(seq 256 16 1024) 1280 1536 1792 2048 | tr ' ' ',')
# Up to 4K additionally appends: 2560,3072,3584,4096
aiperf profile --config configs/default.yaml
aiperf profile --config configs/2k.yaml
aiperf profile --config configs/4k.yaml
```

Configs and launcher: `../qwen38_capture_grid/`. `make fetch-results` copies artifacts locally, `make summarize` collects all six cells into `results/qwen38-capture-grid-20261019/metrics.parquet`, including original AIPerf summary/server/request JSON and the graph-shape histogram. Query with DuckDB directly. No publication or remote Git operations.

## 2026-10-19: additional Qwen3.8 concurrency levels

Added c256, c64, c8 and c1 for all three variants, one fresh server per variant on the same four GPUs and unchanged pinned configuration. Native AIPerf ZipSweep pairs concurrency `[256,64,8,1]` with submission duration `[60,30,30,30]` seconds and 120 s drain grace. c256 gets a longer fill period, producing 32.2 to 39.0 s steady windows. All 12 new cells passed every gate: 4,407 completed requests, zero errors/cancellations/preemptions. The full 18-point dataset has 6,340 completed requests. Driver exited 0, reservation lasted 19m21s, all four GPU workers exited and GPUs were released.

Steady throughput (tok/s), including the preserved c32/c128 cells:

| c | Default | Up to 2K | Up to 4K |
|---:|---:|---:|---:|
| 1 | 412 | 430 | 429 |
| 8 | 1,580 | 1,937 | 1,946 |
| 32 | 2,984 | 4,376 | 4,428 |
| 64 | 3,769 | 6,151 | 6,606 |
| 128 | 4,816 | 8,000 | 9,231 |
| 256 | 7,519 | 8,295 | 11,620 |

At c1, reciprocal mean TPOT is essentially unchanged (~442 tok/s/user), while mean TTFT falls from 119 ms to ~43 ms. At c256, Up to 4K is 40% faster than Up to 2K in the steady window. The smaller Up to 2K benefit at c256 is consistent with mixed-step coverage: one 1,080-token prefill plus 255 decode requests × 4 query tokens totals 2,100 tokens, above 2,048. This is a batch-shape explanation, not a new profiler attribution. Each point remains a single measurement, with different servers for the original and extended levels.

The serving command above is unchanged except for port 8758. Prepare the extension configs with `ROOT="$HOME/runs/qwen38-capture-grid-extra-20261019" PORT=8758 CONCURRENCIES=256,64,8,1 DURATIONS=60,30,30,30 bash run.sh prepare`, then run the same three `aiperf profile --config configs/<variant>.yaml` commands above. Offline validation and expansion passed all 12 configurations before GPU launch.

Canonical combined data: `../qwen38_capture_grid/results/metrics.parquet`, built with DuckDB from both per-suite Parquet files. The original six-cell file remains intact. `make plot` regenerates the expanded PNG/SVG at `results/qwen38-capture-grid-20261019/throughput-interactivity-expanded.*`. The original two-level plot is preserved. Reproduce the extension collection with `make fetch-results RUN_ID=qwen38-capture-grid-extra-20261019`, `make summarize RUN_ID=qwen38-capture-grid-extra-20261019 CONCURRENCIES='256 64 8 1'`, then `make plot`.

## 2026-10-19: Qwen3.8 FP8 deployment recipe, repeated capture-grid comparison

Compared the [B300 recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next?hardware=b300), changing only `cudagraph_capture_sizes`. Same pinned vLLM `58b3298457dde7b4554b3b4e20b238c0ac2c3a65`, 4×B300 TEP4, FP8 weights, MTP3 and prefix caching across arms. Native defaults resolved to context 262,144 and batching budget 16,384. Capture lists are the earlier Default, Up to 2K and Up to 4K lists. The default model runner was used, without the earlier explicit V2 environment override.

- 54/54 measurements passed, 19,411 completed requests, zero errors, cancellations or preemptions. Minimum steady window 10.5 s.
- Three client repeats per grid and concurrency, one server startup per grid. Prefix cache successfully reset before each measurement. SPEED-Bench throughput_1K, shuffle seed 42, natural EOS up to 1,024 output tokens, thinking disabled, temperature 0. Submission 30 s, except c256 at 60 s, grace 120 s.
- [Individual-run plot](../qwen38_capture_grid/results/qwen38-recipe-grid-20261019/throughput-interactivity-expanded.svg): semi-transparent observations, curves connect means. X is the mean across runs of each run's reciprocal mean request TPOT.

Mean steady throughput (tok/s), three measurements per cell:

| c | Default | Up to 2K | Up to 4K | 4K vs Default |
|---:|---:|---:|---:|---:|
| 1 | 376 | 388 | 387 | +3.0% |
| 8 | 1,642 | 2,079 | 2,100 | +27.9% |
| 32 | 3,266 | 4,752 | 4,930 | +51.0% |
| 64 | 4,270 | 6,511 | 7,138 | +67.2% |
| 128 | 4,799 | 8,576 | 9,410 | +96.1% |
| 256 | 7,508 | 7,660 | 11,448 | +52.5% |

c256 ranges are 7,295–7,811 / 6,944–8,066 / 11,318–11,581 tok/s. Up to 2K has little consistent c256 benefit, while Up to 4K clearly improves this tested workload. c1 TPOT is 2.45 / 2.49 / 2.50 ms, so its small throughput gain does not imply improved decode interactivity.

| Grid | KV tokens | Actual graph GiB/GPU | Capture s | Startup s |
|---|---:|---:|---:|---:|
| Default | 26,640,915 | 2.85 | 38 | 245 |
| Up to 2K | 26,683,425 | 2.68 | 37 | 250 |
| Up to 4K | 26,531,806 | 4.04 | 39 | 229 |

Up to 4K loses 109,109 KV tokens (0.41%), adds 1.19 GiB/GPU graph memory and 1 s capture time versus Default. These are single startup observations per grid. The small 2K memory reversal and total startup differences should not be treated as causal improvements, allocation and cache state vary between starts. These measurements support a model/deployment recipe option, not a universal default change.

The saved graph logs explain the 2K discontinuity. In Up to 2K steady-state samples, among steps above 1,024 tokens, c128 has 248/306 (81.0%) within 2K, 51/306 (16.7%) between 2K and 4K and 7/306 (2.3%) above 4K. At c256, these are 23/449 (5.1%), 327/449 (72.8%) and 99/449 (22.0%). All sampled `NONE` steps exceed the configured cap. These are step-count fractions, not runtime fractions. Samples cover three complete 10-second log intervals at c128 and six at c256, wholly within the measured steady windows, with a one-second guard for timestamp rounding. Default c128 has no complete interval inside its shorter steady windows and is omitted from this log-window comparison.

With MTP3, each continuing decode request contributes four query tokens. For example, a 1,080-token prefill plus 127 decode requests gives 1,588 tokens at c128, while the same prefill plus 255 decode requests gives 2,100 at c256. Thus the latter misses the 2K grid. This shape-coverage evidence supports the threshold explanation, without assigning wall-time shares to eager steps. Reproduce with `duckdb -csv -f graph-coverage.sql` in `../qwen38_capture_grid`. The extracted shapes are in `results/qwen38-recipe-grid-20261019/steady-graph-shapes.parquet`.

Graph-enabled execution steps (`FULL` + `PIECEWISE`), pooled by step count across the three repeats:

| c | Default | Up to 2K | Up to 4K |
|---:|---:|---:|---:|
| 1 | 99.75% | 100.00% | 100.00% |
| 8 | 96.29% | 99.93% | 100.00% |
| 32 | 87.92% | 99.07% | 100.00% |
| 64 | 81.83% | 96.84% | 100.00% |
| 128 | 79.69% | 94.07% | 99.52% |
| 256 | 49.94% | 55.23% | 92.86% |

This table samples complete 10-second logging intervals within each client's first-request to last-completion window, including filling and draining, rather than the narrower steady window used above. All 54 runs contribute. It measures execution-step coverage, not request coverage or time coverage. A request can span multiple dispatch modes, and `PIECEWISE` retains eager sections. The same SQL exports `graph-coverage.csv` and `graph-coverage-runs.csv` alongside the recipe metrics.

Portable serving command, equivalent to the pinned local checkpoint used by the runner:

```bash
env -u VLLM_USE_V2_MODEL_RUNNER VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3.8-Flash-Next-FP8 --revision 236dfdf285828023ca3bcd3f37366c58a3469b13 --served-model-name qwen38 --tensor-parallel-size 4 --enable-expert-parallel --gpu-memory-utilization 0.90 --max-num-seqs 256 --enable-prefix-caching --kv-cache-dtype fp8 --attention-config.indexer_kv_dtype fp8 --speculative-config '{"method":"mtp","num_speculative_tokens":3}' --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --disable-uvicorn-access-log --cudagraph-metrics --host 127.0.0.1 --port 8768
# Up to 2K / Up to 4K append the earlier complete capture list:
# --compilation-config "{\"cudagraph_capture_sizes\":[$SIZES]}"
curl --fail --request POST http://127.0.0.1:8768/reset_prefix_cache
aiperf profile --config configs/<grid>/r<repeat>-c<concurrency>.yaml
```

Development endpoints were enabled on loopback for configuration inspection and cache resets, identically across arms. First attempt failed before measurements because those endpoints were disabled. The corrected attempt completed all 54 cells and released GPUs 0–3. Canonical data: `../qwen38_capture_grid/results/qwen38-recipe-grid-20261019/metrics.parquet`, with original AIPerf summary, server metrics and request records retained. Rebuild from `../qwen38_capture_grid` with `make fetch-results RUN_ID=qwen38-recipe-grid-20261019`, `make summarize RUN_ID=qwen38-recipe-grid-20261019 CONCURRENCIES='1 8 32 64 128 256'`, then `make recipe-plot RUN_ID=qwen38-recipe-grid-20261019`. Results remain local, nothing published or pushed.

## 2026-10-19: measured cost-curve experiment (rh-a8g)

Measured cost curves change AV's choices and improve c128 throughput by **10.45%** over fresh startup-table AV. They regress c1 by 2.20% and c8 by 3.74%, so this table does not satisfy the criterion of matching the best fixed k everywhere. At c128, C is 10,218 [9,990, 10,234] tok/s versus B's 9,252 [9,229, 9,486]. Its advantage over fixed k=5, 10,060 [9,877, 10,088], is small relative to repeat spread.

All 168 collected cells passed the existing gates, including 75 main A/B/C/D comparisons. Prompt-set hashes match at every c. B, C and D use GPUs 6/7 with three untraced repeats per arm. A reuses the prior extended-graph run on GPUs 0/1. Extra C repeats on GPUs 4/5 and a separate trace are retained as validation. The earlier B is preserved as `B-prior`, fresh B was added because the traced startup-table control differed materially at c128.

Each cell below is **tok/s / TPOT ms / AL**, medians of three repeats. The [full comparison CSV](results/cost-curve-analysis-20261019/comparison.csv) includes min/max for every metric.

| c | A: off k=7 | B: startup AV | C: measured AV | D: off k=3 | D: off k=5 |
|---|---|---|---|---|---|
| 1 | 488 / 1.85 / 2.708 | 494 / 1.86 / 2.605 | 483 / 1.93 / 2.491 | 498 / 1.87 / 2.393 | 505 / 1.82 / 2.599 |
| 8 | 2,234 / 3.34 / 2.991 | 2,245 / 3.30 / 2.789 | 2,161 / 3.42 / 2.775 | 2,190 / 3.40 / 2.547 | 2,254 / 3.30 / 2.872 |
| 32 | 4,994 / 5.52 / 2.951 | 4,954 / 5.47 / 2.667 | 5,033 / 5.47 / 2.854 | 4,798 / 5.83 / 2.548 | 4,981 / 5.57 / 2.844 |
| 64 | 7,118 / 7.74 / 2.926 | 7,196 / 7.51 / 2.527 | 7,334 / 7.34 / 2.775 | 7,106 / 7.96 / 2.503 | 7,216 / 7.64 / 2.786 |
| 128 | 9,330 / 11.05 / 2.942 | 9,252 / 10.92 / 2.401 | 10,218 / 10.02 / 2.755 | 9,990 / 10.62 / 2.531 | 10,060 / 10.46 / 2.830 |

[Throughput/TPOT/AL figure](results/cost-curve-analysis-20261019/cost-experiment.svg), lines are medians and shading is min/max. [Measured versus startup cost curves](results/cost-curve-analysis-20261019/calibrated-costs.svg) uses the startup profile from the separate traced B control.

Mean executed drafts per steady, full-concurrency, decode-only step, one traced run per policy:

| c | B | C | C steps verifying all eligible drafts |
|---|---|---|---|
| 1 | 4.16 | 3.13 | 7.50% |
| 8 | 24.22 | 24.61 | 0% |
| 32 | 73.81 | 95.21 | 0% |
| 64 | 121.63 | 171.79 | 0% |
| 128 | 128.00 | 293.69 | 0% |

Calibration used k=1/2/3/5/7 at R=1/8/32/64/128, decode-only start-to-next-start gaps, medians per exact executed shape. These gaps include drafting and CPU overhead. The fit uses monotone request intercepts plus a shared concave token-cost curve. RMSE across 25 shape medians is 0.374 ms, maximum residual 1.511 ms. A frozen k=1/2/3/5 fit predicted held-out k=7 shapes with 0.857 ms RMSE. [Coefficients and dense decode curves](results/cost-curve-analysis-20261019/cost-curves.json), [held-out validation](results/cost-curve-analysis-20261019/heldout-validation.json), [injection patch](av-cost-curves.patch).

The patch supplies all table indices and retains startup verify-cost increments above 1,024 tokens. The trace verified all 513 draft entries and the calibrated verify prefix, with 16,385 verify entries in total. The configured piecewise capture list reaches 6,144 tokens, the profiled full-graph limit is 4,096. Curve SHA-256: `545245792131dd8102fb60066f3f62bc3b134b230e618ad6f56cdd69cf2f5986`.

The next calibration should hold drafter k=7 fixed, tracked in **rh-2bm**. DFlash's `num_query_per_req = 1 + num_speculative_steps`, so the fixed-k calibration also changes drafter work. This experiment establishes that pricing changes can improve high-concurrency AV, while a clean separation of drafter and verification costs remains necessary before attributing the low-concurrency regression to AV's objective or acceptance estimate.

Representative C server command, run outside the checkout under the harness's `canhazgpu run` reservation:

```bash
CG_SIZES=$(printf '%s\n' 1 2 4 $(seq 8 8 248) $(seq 256 16 1024) $(seq 1280 256 4096) 5120 6144 | paste -sd,)
PYTHONPATH="$HOME/code/vllm-cost-corrected-untraced-60229" VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=1024 VLLM_AV_COST_CURVES="$HOME/runs/qwen-cost-matched-20261019/cost-curves.json" ~/.venv/bin/python -m vllm.entrypoints.openai.api_server --model /data/tomasruizt/hub_cache/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754 --served-model-name qwen --language-model-only --max-model-len 16384 --max-num-seqs 512 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.92 --no-enable-prefix-caching --disable-uvicorn-access-log --host 127.0.0.1 --port 8678 --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.5-35B-A3B-DFlash","revision":"52cb554b4995dede3e2e1bdb129cdb1f3529332b","num_speculative_tokens":7,"draft_sample_method":"probabilistic","enable_adaptive_verification":true}' --compilation-config "{\"cudagraph_capture_sizes\":[$CG_SIZES]}"
sed -e 's/__PORT__/8678/g' -e 's/__C__/128/' -e 's/__N__/512/' steady-speedbench.yaml > c128.yaml
QUICK_ARTIFACTS=artifacts-c128 timeout 900 ~/aiperf-venv/bin/aiperf profile --config c128.yaml
```

All arms use `1:24 8:96 32:192 64:320 128:512 1:24` cells. Rebuild locally with `make fetch-cost-results`, `make fit-cost-curves`, then `make summarize-cost`. The last target runs DuckDB directly through `cost_analysis.sql`. The common dataset is `results/cost-curve-analysis-20261019/metrics.parquet` (168 rows, original AIPerf JSON retained), with `steps.parquet`, `budgets.parquet`, `decisions.csv` and `cost-transfer.csv` alongside it. The original PR checkout remained unchanged, experiment changes and results are local.

## 2026-10-19: q1/q2 dip is skipped GDN state computation (rh-6ex)

Confirmed on exact PR `888ec6bb5745e6060fa2d6a424a37bebbbd84cb9`, Qwen3.5-35B-A3B BF16, DFlash k=7, B300, context 1,024. The q2 advantage disappears when dummy GDN states are valid. This supersedes the unverified q1 slow-path explanation above.

| Exact shape | Original dummy states, target ms | Valid distinct states, target ms | Original / valid GDN kernel ms |
|---|---:|---:|---:|
| 128×1 | 7.180 | 7.031 | 2.794 / 2.816 |
| 128×2 | 4.758 | 8.546 | 0.257 / 4.043 |
| 128×8 | 8.092 | 19.998 | 0.281 / 12.356 |
| 512×1 | 19.506 | 19.296 | 11.203 / 11.225 |
| 512×2 | 10.517 | 25.909 | 0.891 / 16.411 |

Target times are median CUDA-event measurements from three groups of 50 exact outer-graph replays, without the drafter, sampling or scheduler. GDN columns sum CUDA kernel durations across 30 layers in three profiler replays, divided by three. Profiler durations are perturbed and should not be subtracted directly from unprofiled target times. One model initialization for the completed comparison, synthetic dummy inputs, no accuracy evaluation or independent server repeats. The unmodified startup timing control separately reproduced 512×1 = 19.439 ms and 512×2 = 10.388 ms (three calls each).

Root cause, with line numbers in that PR:

- `vllm/v1/worker/gpu/input_batch.py:263-279`: `set_dummy_context` fills block-table columns according to context length. Mamba's 16,384-token block size needs only column 0 at context 1,024. Its table becomes `[request_index, 0, ..., 0]`, including a null first request.
- `vllm/v1/attention/backends/gdn_attn.py:705-719`: capture metadata sets `num_accepted_tokens = diff(query_start_loc)` and keeps AV rows on the spec path. q1 selects column 0, q2 selects column 1, q8 selects column 7. Recorded metadata confirms accepted counts 1/2/8 and these zero columns in all 30 GDN layers.
- `csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu:174-189`: `gdn_decode_post_conv_mtp_kernel` selects `state_indices[accepted - 1]`, zeroes output and returns for `source_slot <= 0`. q2/q8 skip recurrent state loads, updates and stores. This is the same `<float, 2, false>` kernel in all four 512-request controls, 30 calls, grid `[512,32,1]`, block `[256,1,1]`, 71 registers/thread, 45,120 shared bytes. The dip is not a q1/q2 kernel dispatch or grid change.

An independent single-kernel control, with fixed BF16 QKV/gates and FP32 recurrent state, changes only the selected source slot between valid and zero. Median of three groups of 100 CUDA-graph replays on GPU 5:

| Requests | Valid q1 ms | Valid q2 ms | Null-source q1 ms | Null-source q2 ms |
|---|---:|---:|---:|---:|
| 128 | 0.0984 | 0.1362 | 0.0084 | 0.0102 |
| 512 | 0.3745 | 0.5273 | 0.0267 | 0.0287 |

Valid q2 is 21.5% slower at 128 requests and 34.3% slower at 512 requests for the complete target forward. There is no demonstrated serving gain from padding q1 to q2. Normal serving reserves block 0 as null (`vllm/v1/core/block_pool.py:184-188`) and allocates speculative Mamba blocks (`single_type_kv_cache_manager.py:1824-1835`). Plain non-spec decode has its own packed GDN path. This investigation does not benchmark plain serving or propose changing its routing, following the bounded shape-check scope on rh-6ex.

Implemented `probe_query_shapes.py`, `probe_gdn_state_slots.py`, `run-query-shapes.sh`, `analyze_query_shapes.py`, and Makefile fetch/analysis targets. The exact-shape probe assigns distinct nonzero slots for every accessed query-state column and checks pool capacity. Recommendation: fix AV's dummy Mamba allocation and validate selected sources/destinations before profiling, then revisit request-count-aware calibration. Production fix tracked in **rh-5u9**, including finite state-pool handling. Do not change real accepted-state offsets to hide the profiling defect.

Commands, run asynchronously with `nohup ... > run.log 2>&1 < /dev/null &` from outside the checkout:

```bash
chg run --gpu-ids 4 --timeout 45m --note "rh-6ex exact q1 q2 target forwards" -- bash "$HOME/runs/qwen-dflash-av-60229/run-query-shapes.sh"
chg run --gpu-ids 5 --timeout 10m --note "rh-6ex isolated GDN state-slot control" -- env PYTHONPATH="$HOME/code/vllm-validate-60229" PROBE_OUTPUT="$HOME/runs/qwen-q1q2-20261019" "$HOME/.venv/bin/python" "$HOME/runs/qwen-dflash-av-60229/probe_gdn_state_slots.py"
make fetch-query-results
make analyze-query-shapes
duckdb -c "SELECT requests, query, valid_states, median(gpu_ms) AS ms FROM read_parquet('results/qwen-q1q2-20261019/metrics.parquet') WHERE scope='target_forward' GROUP BY ALL ORDER BY requests,query,valid_states"
```

`run-query-shapes.sh` records the exact model path, PR `PYTHONPATH` and environment. `probe_query_shapes.py` contains the complete LLM configuration, including pinned drafter revision and the original 0.92 memory fraction, 16K context/token budgets and 512 sequence limit. Results: `results/qwen-q1q2-20261019/metrics.parquet` (66 measurements: 30 full-forward, 24 isolated-kernel, 12 startup controls), `kernels.parquet` (340 kernel groups), per-shape metadata and Chrome traces. Both jobs finished, GPUs 4/5 were released, validated checkout unchanged. No publishing or upstream changes.

## 2026-10-19: production state-slot fix and raw startup curves (rh-5u9)

Draft for code review: https://github.com/tomasruizt/vllm/pull/21, fork branch `fix/av-gdn-profile-states`, commit `cde317f12e`, base snapshot `888ec6bb57` (`av-gdn-profile-base`). Three-file diff: populate Mamba query-state columns with bounded nonzero IDs in `set_dummy_context`, pass group IDs from the runner, add metadata regression coverage. Existing dummy block reuse beyond pool capacity is retained. Real request allocation and accepted-state offsets are unchanged.

Native raw verify-cost medians before clamping, same B300 GPU 4 sequentially, original model/configuration, one initialization per arm, five profiling samples per point:

| Target tokens | Base ms | Fixed ms |
|---|---:|---:|
| 128 | 7.183 | 7.225 |
| 512 (512×1) | 19.419 | 19.420 |
| 768 (mixed q1/q2) | 15.392 | 23.271 |
| 1,024 (512×2) | 10.357 | 27.140 |

The q1→q2 inversion disappears. Tiny local fluctuations remain, so monotonic timing is not a unit-test assertion. The eager tail still drops at 4,096→8,192 (71.767→45.224 ms on the branch). Those dummy rows exceed the GDN MTP width and need a separate schedulable-tail fix, tracked in **rh-cg4**. No claim that the entire startup curve or serving decisions are now correct.

Validation: **115 passed** across `test_gpu_input_batch_v2.py` and `test_gpu_pcp_manager.py`, pre-commit passed including mypy and test tethering. New CPU metadata cases cover q1/q2/q8, small/normal pools and aligned context offsets. Neutralizing Mamba-specific population yields **11 failed, 1 passed**. GPUs 4/5 released, original checkout unchanged. No model accuracy or serving throughput evaluation.

Reproduce the paired native profile with `chg run --gpu-ids 4 --timeout 30m --note "rh-5u9 raw AV cost curve base versus fixed" -- bash "$HOME/runs/qwen-dflash-av-60229/run-raw-cost-curves.sh"`. The runner contains all environment/configuration settings and calls `profile_raw_cost_curve.py`, which intercepts `set_cost_curves` before clamping. Fetch/analyze with `make fetch-raw-cost-curves` and `make analyze-raw-cost-curves`. Results: `results/qwen-gdn-profile-fix-20261019/metrics.parquet` (278 median points across base/fixed, draft/verify), [raw curve plot](results/qwen-gdn-profile-fix-20261019/raw-cost-curves.png), SVG alongside it. No logs or raw data were published with the draft.

## 2026-10-19: valid mixed profiling tails (rh-cg4)

Implemented in `7ff73c5a20`, added to the same [fork draft](https://github.com/tomasruizt/vllm/pull/21). No hard-coded GDN width: AV marks points above configured decode capacity as mixed, then the runner reserves prefill request slots and fills the surplus with prompt chunks. The explicit split, prefill flags, zero context for fresh prefills and physical graph padding survive dummy construction. Mixed batches cannot select decode-only FULL graphs. Existing dummy block reuse beyond cache capacity remains.

The old tail guard is now confirmed by torch traces: at 4K/8K/16K the state-slot-only version labels all 512 requests as q8/q16/q32 speculative decode, with non-null sources. Thirty fused GDN MTP calls total 49.621/1.253/1.683 ms. Thus the oversized-query guard, rather than missing states, skips recurrent work in the tail.

Native raw verification costs before clamping, paired sequentially on GPU 4, same Qwen3.5/DFlash configuration, one initialization per arm and five samples per point:

| Target tokens | State slots fixed only, ms | Mixed tail, ms |
|---|---:|---:|
| 4,096 | 71.844 | 71.861 |
| 8,192 | 45.348 | 115.188 |
| 16,384 | 79.774 | 160.690 |

Actual mixed metadata at 8K is 507×8 decode tokens plus 4,136 prefill tokens across five requests. At 16K it is 499×8 plus 12,392 prefill tokens across thirteen requests. All 30 GDN layers report the corresponding decode/prefill counts and selected sources are non-null. At 8K, traces contain 30 `fused_sigmoid_gating_delta_rule_update_kernel` calls (58.350 ms) and 30 FlashInfer `GatedDeltaNetChunkedKernel` calls (2.370 ms). The invalid-query early return is gone.

All-prefill controls, one fresh prompt, median of three unprofiled probes: 4K/8K/16K = 63.829/63.743/108.381 ms. These differ substantially from the mixed costs, so the curve remains a representative policy rather than a universal function of token count. Small local decreases remain in the raw curve. No monotonic timing assertion, model accuracy evaluation or serving throughput claim.

Validation: **201 passed** (152 AV/input-batch/runner/PCP tests, 49 CUDA-graph tests), pre-commit including mypy/test tethering passed. Metadata tests cover multiple configured widths, request budgets, fresh prefill context and physical padding. Existing upstream mixed-tail work was inspected as a reference. Its explicit split alone does not supply prefill classification/context on this frozen AV base. No duplicate upstream submission was opened.

Reproduce with `chg run --gpu-ids 4 --timeout 35m --note "rh-cg4 native tail curves and GDN traces" -- bash "$HOME/runs/qwen-dflash-av-60229/run-tail-profiles.sh"`, then `make fetch-tail-profiles` and `make analyze-tail-profiles`. Full model settings and both checkout paths are in the runner and `profile_tail_batches.py`. Canonical dataset: `results/qwen-av-tail-profile-20261019/metrics.parquet` (203 rows, filter `scope='native_curve'` or `scope='probe'`), `kernels.parquet` (928 groups), per-shape metadata/Chrome traces and [before/after plot](results/qwen-av-tail-profile-20261019/mixed-tail-cost-curves.png). Only the checked PNG was published for embedding in the draft. GPUs 4/5 released, original validated checkout unchanged.

## 2026-10-19: cost calibration with drafter k=7 held fixed (rh-2bm)

Revised costs improve c128 throughput **7.10%** over fresh startup AV, but regress c1/c8 by **2.65%/3.81%**. The c128 advantage over fixed k=5 is 1.44%, with overlapping repeat ranges. Removing the changing-drafter-k confound improves cost prediction, but does not produce a universal serving win.

Calibration now keeps DFlash at k=7 and uses AV's normal global compaction with total decode verification budgets `min(available, R × {1,3,7})`. Prefill follows normal AV. All 18 calibration cells pass validity gates, prompt hashes match, and all 68,538 checked decode-budget records match the intervention. This uses frozen PR commit `888ec6bb5745e6060fa2d6a424a37bebbbd84cb9`, independently of the profiling fix above.

Across 15 full-concurrency shapes at R=1/8/32/64/128, the revised concave fit has **0.113 ms RMSE**, maximum residual 0.230 ms. The prior empirical table has 2.459 ms RMSE on these measurements. Holding the three-draft shapes out gives 0.308 ms RMSE for the concave fit versus 1.009 ms for a monotone fit, so the concave fit was selected before serving comparisons. These are step-start gaps including target, drafter and CPU overhead, the request-indexed component is not a standalone drafter timing.

[Measured and startup costs](results/fixed-drafter-analysis-20261019/calibrated-costs.svg), [dense curves and coefficients](results/fixed-drafter-analysis-20261019/cost-curves.json), [held-out validation](results/fixed-drafter-analysis-20261019/fit-validation.json), [prior-table validation](results/fixed-drafter-analysis-20261019/prior-curve-validation.json). Frozen curve SHA-256: `3292eca14065f79d75a1c2369efc88a38c6eb24a5b17c83b14182fe776738553`.

All **62 cells passed** the validity gates, with one prompt-set hash per c. B/C/D each have three untraced repeats, one on GPU 6 and two on GPU 7, plus separate B/C decision traces on GPU 6. Capture sizes and AIPerf settings are unchanged. Maximum absolute c1 sentinel drift is 1.97%. Original checkout unchanged, both GPU reservations released.

Each cell is **tok/s / TPOT ms / AL**, medians of three repeats. [Comparison CSV](results/fixed-drafter-analysis-20261019/comparison.csv) contains min/max for every metric, [figure](results/fixed-drafter-analysis-20261019/cost-experiment.svg).

| c | B: startup AV | C: revised-cost AV | D: fixed k=5 |
|---|---|---|---|
| 1 | 486 / 1.89 / 2.585 | 473 / 1.95 / 2.522 | 497 / 1.85 / 2.605 |
| 8 | 2,227 / 3.31 / 2.839 | 2,142 / 3.45 / 2.773 | 2,219 / 3.35 / 2.839 |
| 128 | 8,930 / 11.41 / 2.570 | 9,565 / 10.79 / 2.801 | 9,429 / 11.21 / 2.831 |

Mean verified drafts per request in steady, full-concurrency, decode-only steps, one trace per policy:

| c | B | C | C verifies all eligible drafts |
|---|---|---|---|
| 1 | 4.65 | 3.24 | 8.41% |
| 8 | 3.03 | 3.12 | 0% |
| 128 | 1.61 | 2.54 | 0% |

[Decision CSV](results/fixed-drafter-analysis-20261019/decisions.csv), [cost transfer](results/fixed-drafter-analysis-20261019/cost-transfer.csv), [measured costs per executed shape](results/fixed-drafter-analysis-20261019/runtime-shapes.csv). Median C step residuals against the frozen fit are -0.051/+0.840/+0.634 ms at c1/c8/c128, so training RMSE is not a serving-transfer guarantee. The C trace verifies all 513 draft entries and the 1,025-entry calibrated verification prefix, with 16,385 verification entries overall and full-graph limit 4,096.

Low aggregate fit error does not guarantee correct marginal prices. At R=1, T=2/4/8 costs 4.779/4.741/5.736 ms, while the selected table predicts 4.626/4.970/5.659 ms. It invents a 0.345 ms penalty from one to three verified drafts where the controlled measurements are nearly flat. This low-R shape is incompatible with concavity. Real traces also show variation within a capture bucket, so snapping all prices to capture boundaries is not sufficient evidence of correctness. More controlled small-budget measurements are tracked in **rh-b60**, before attributing the remaining gap to AV's objective or confidence estimates.

Representative C serving and benchmark commands, outside the checkout under the harness reservation:

```bash
CG_SIZES=$(printf '%s\n' 1 2 4 $(seq 8 8 248) $(seq 256 16 1024) $(seq 1280 256 4096) 5120 6144 | paste -sd,)
PYTHONPATH="$HOME/code/vllm-fixed-drafter-controls-60229" VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=1024 VLLM_AV_COST_CURVES="$HOME/runs/qwen-fixed-drafter-controls-20261019/cost-curves.json" ~/.venv/bin/python -m vllm.entrypoints.openai.api_server --model /data/tomasruizt/hub_cache/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754 --served-model-name qwen --language-model-only --max-model-len 16384 --max-num-seqs 512 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.92 --no-enable-prefix-caching --disable-uvicorn-access-log --host 127.0.0.1 --port 8683 --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.5-35B-A3B-DFlash","revision":"52cb554b4995dede3e2e1bdb129cdb1f3529332b","num_speculative_tokens":7,"draft_sample_method":"probabilistic","enable_adaptive_verification":true}' --compilation-config "{\"cudagraph_capture_sizes\":[$CG_SIZES]}"
sed -e 's/__PORT__/8683/g' -e 's/__C__/128/' -e 's/__N__/512/' steady-speedbench.yaml > c128.yaml
QUICK_ARTIFACTS=artifacts-c128 timeout 900 ~/aiperf-venv/bin/aiperf profile --config c128.yaml
```

[Launch plan](run-fixed-drafter-controls.sh), [budget-control patch](av-fixed-drafter-budget.patch). Rebuild with `make fetch-fixed-drafter-results`, `make fit-fixed-drafter`, `make summarize-fixed-drafter`. Canonical data: `results/fixed-drafter-analysis-20261019/metrics.parquet` (62 rows, raw AIPerf metrics retained), `steps.parquet` and `budgets.parquet`. Python/shell syntax checks and patch application against the frozen source passed. Changes remain local, nothing published.

## 2026-10-19: measured-cost oracle, acceptance estimation unchanged (rh-b60, rh-1yq)

Replacing the cost denominator with directly measured means, while preserving acceptance estimation and confidence ranking, gives **+3.08% / -0.24% / +2.17%** median paired throughput changes at c1/c8/c128. The c128 result covers **36.61% of steady choices**, so the full perfect-cost question remains open there.

Six paired untraced runs per c, three per GPU, with pair order reversed between GPUs. Cells show **tok/s / TPOT ms / AL**, each a median. Gains are medians of paired ratios, their observed ranges include both signs. Coverage comes from separate decision traces during the same steady-window definition.

| c | Native AV | Measured costs | Median paired gain (range) | Priced steady choices |
|---|---|---|---|---|
| 1 | 499 / 1.849 / 2.612 | 510 / 1.810 / 2.560 | +3.08% (-2.67% to +4.11%) | 100% |
| 8 | 2,245 / 3.298 / 2.809 | 2,246 / 3.279 / 2.719 | -0.24% (-1.42% to +2.57%) | 96.12% |
| 128 | 9,523 / 10.696 / 2.440 | 9,807 / 10.415 / 2.594 | +2.17% (-0.24% to +5.35%) | 36.61% |

[Paired runs and medians](results/cost-oracle-policy-analysis-20261019/paired-gains.svg), [comparison CSV](results/cost-oracle-policy-analysis-20261019/comparison.csv), [all paired ratios](results/cost-oracle-policy-analysis-20261019/pairs.csv). Standard AIPerf throughput gives corresponding paired changes +3.01% / -0.11% / +2.82%. All 48 cells pass, 36 untraced comparisons plus 12 separate diagnostics, with one prompt-set hash per c. The unchanged workload enables EOS with a 1,024-token cap, native runs also vary in generated text.

Mean verified drafts per request in steady full-concurrency decode traces change **4.544→3.541 / 3.359→2.821 / 1.318→2.332** at c1/c8/c128. All actual budgets match compaction, native budgets remain unchanged, and every applied oracle budget matches the original argmax with the measured denominator. [Decision audit](results/cost-oracle-policy-analysis-20261019/decisions.csv). Median absolute price errors on the oracle trajectory are 0.08–0.15 / 0.38–0.40 / 0.61 ms at R=1/8/128, including trace overhead. [Cost transfer](results/cost-oracle-policy-analysis-20261019/cost-transfer.csv), [replay using unchanged estimated acceptance prefixes](results/cost-oracle-policy-analysis-20261019/frozen-state-objective.csv).

The remaining c128 gap is R=124–127 decode and mixed/prefill steps with about 1,200 non-draft tokens. Native cost fallbacks leave the full perfect-cost benefit unresolved, further measurements are tracked in **rh-dte**. Both workers exited successfully and released their GPUs, original commit `888ec6bb` unchanged. Canonical serving data: `results/cost-oracle-policy-analysis-20261019/{metrics,pairs,decisions}.parquet`, including AIPerf metrics and outputs.

Cost collection randomizes every integer global budget `b=0..7R` at R=1/8/32/64/128, keeping drafter k=7 and normal confidence ranking. Two folds retain real contexts, draft layouts and AIPerf metrics. Costs are direct means/medians without fitting, interpolation or separability assumptions. Independent native-policy validation uses the other fold's costs, excluding pooled estimates.

**Timing correction:** gap `t+1` prices batch `t`, AV waits on the preceding confidence-copy event. [Analysis](cost_oracle.sql) preserves raw traces and applies this shift with three stable decode neighbors without prefill. Earlier native per-shape residuals in `runtime-shapes.csv` require this correction. Constant-budget calibration and untraced throughput results remain unchanged.

Frozen [lookup](results/cost-oracle-analysis-20261019/cost-lookup-frozen.json), [measured curves](results/cost-oracle-analysis-20261019/measured-costs.svg), [coverage](results/cost-oracle-analysis-20261019/coverage.csv), [independent validation](results/cost-oracle-analysis-20261019/validation.csv). All **1,636 primary budget points** are observed, with 70,941 usable measurements. Minimum pooled samples at R=1/8/32/64/128: **2,790/90/36/26/7**. Only 12 points have fewer than ten samples. SHA-256: `c513373952be1f35163a6a5df72feabc220908866f8119fe8ddc1d2af92a7791`.

All 38 natural-EOS collection cells pass serving gates. The 120 fixed-length calibration cells pass completion/error/preemption/queue checks, their cohort windows are excluded from throughput comparisons. Zero eligible budget mismatches. Canonical data: `results/cost-oracle-analysis-20261019/{raw-steps,steps,points,validation,metrics}.parquet`. Rebuild with `make fetch-cost-oracle-results` and `make analyze-cost-oracle`. [79-cell plan](cost-oracle-plan.json), [runner](run-cost-oracle.sh), [patch](av-cost-oracle.patch). Collection workers exited successfully, nothing published.

Representative serving command inside a `canhazgpu run` reservation, native and measured-cost cells reuse this server:

```bash
ROOT="$HOME/runs/qwen-av-cost-oracle-policy-a-20261019"
mkdir -p "$ROOT"
printf '{"mode":"native","phase":"startup"}\n' > "$ROOT/control.json"
CG_SIZES=$(printf '%s\n' 1 2 4 $(seq 8 8 248) $(seq 256 16 1024) $(seq 1280 256 4096) 5120 6144 | paste -sd,)
PATH="$HOME/.local/bin:$HOME/.venv/bin:$PATH" PYTHONPATH="$HOME/code/vllm-cost-oracle-policy-60229" AV_COST_CONTROL="$ROOT/control.json" AV_COST_TRACE="$ROOT/cost-steps.jsonl" AV_COST_LOOKUP="$HOME/runs/qwen-cost-oracle-tools-20261019/cost-lookup-frozen.json" AV_COST_MIN_SAMPLES=5 AV_COST_SEED=42 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=1024 VLLM_AV_COST_CURVES= ~/.venv/bin/python -m vllm.entrypoints.openai.api_server --model /data/tomasruizt/hub_cache/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754 --served-model-name qwen --language-model-only --max-model-len 16384 --max-num-seqs 512 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.92 --no-enable-prefix-caching --disable-uvicorn-access-log --host 127.0.0.1 --port 8686 --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.5-35B-A3B-DFlash","revision":"52cb554b4995dede3e2e1bdb129cdb1f3529332b","num_speculative_tokens":7,"draft_sample_method":"probabilistic","enable_adaptive_verification":true}' --compilation-config "{\"cudagraph_capture_sizes\":[$CG_SIZES]}"
```

Representative measured-cost client command, with the unchanged [AIPerf configuration](steady-speedbench.yaml), EOS enabled and output limit 1,024:

```bash
ROOT="$HOME/runs/qwen-av-cost-oracle-policy-a-20261019"
printf '{"mode":"oracle","phase":"manual-c128","trace":false}\n' > "$ROOT/control.json"
sed -e 's/__PORT__/8686/g' -e 's/__C__/128/' -e 's/__N__/512/' steady-speedbench.yaml > c128.yaml
QUICK_ARTIFACTS=artifacts-c128 timeout 1800 ~/aiperf-venv/bin/aiperf profile --config c128.yaml
```

[Cost-only policy patch](av-cost-oracle-policy.patch), [paired plan](cost-oracle-policy-plan.json), [lookup policy](av_cost_oracle_policy.py). Rebuild serving comparisons with `make fetch-cost-oracle-policy-results` and `make summarize-cost-oracle-policy`.

## 2026-10-19: GDN-only follow-up on community profiling shapes (rh-4by)

Fork draft now compares `av-community-gdn-base` (`65d3336435`) against `fix/av-gdn-profile-states` (`bd45e45232`, local branch `fix/av-gdn-on-community`). The base contains community head `66821828df` plus the existing GDN AV support prerequisite, which is required to run Qwen with AV. Only three files differ: dummy GDN state allocation, prefill metadata/context and regression tests. Community batch splitting is unchanged. The former implementation is preserved on `archive/av-gdn-before-community`.

Native target costs before clamping, community / community plus GDN fixes in ms: 512 tokens **19.150 / 20.222**, 1,024 **10.212 / 25.158**, 4,096 **54.531 / 68.868**, 8,192 **98.029 / 112.892**, 16,384 **145.800 / 159.974**. Sequential runs on B300 GPU 4, one initialization per arm, five samples per point, same pinned Qwen/DFlash weights and 1,024 context. Each final arm has 88 verification and 51 draft points. Discard preliminary memory-profile curves recorded before final graph capture.

At 1,024 tokens, community metadata has 512 null GDN sources, fixed metadata has zero. The current community branch already executes prefill kernels in the tail. Both 16K splits remain 500×8 decode tokens plus 12,384 prefill tokens across 12 requests. The follow-up makes the dummy flags and fresh-prefill context consistent with that split. The recreated [diagram](plots/community-gdn-preview/serving-cost-comparison.png) uses the community baseline in red and the GDN follow-up in teal, with shared y-axis and unclamped costs. Historical serving periods remain contextual, not a matched accuracy or throughput evaluation.

Validation: **180 tests passed**, changed-file pre-commit passed, tested source hashes match the pushed branch. Both managed GPU jobs completed and GPUs 4/5 were released. Rebuild with `make fetch-community-gdn` and `make analyze-community-gdn`. Canonical long-format data: `results/qwen-gdn-community-20261019/metrics.parquet` (63,410 rows). The user-approved simplified diagram was embedded in the draft on 2026-10-20, with measurement provenance and limitations in the description.

Readability cleanup on 2026-10-20 (`9327f59897`, rh-4fy): separated dummy Mamba metadata from context setup, flattened state allocation and removed the nested group scan. **188 CPU tests passed**, nine skipped and eight CUDA-only cases deselected. All 384 before/after batch and state-table comparisons match exactly, and pre-commit passed. The diagram data is unchanged. The earlier 180-test GPU validation was not repeated while the GPUs were occupied.

## 2026-10-20: c256 with max-num-seqs 256, empirical costs (rh-fll)

**No measured throughput improvement:** six paired comparisons give **−1.08%** median steady throughput change (−1.78% to +0.89%). Whole-profile AIPerf gives −0.31%. This experiment uses TP2, the previous c128 experiment used one GPU.

| GPU pair | Native tok/s / TPOT ms / AL | Measured costs tok/s / TPOT ms / AL |
|---|---|---|
| A | 17,879 / 11.539 / 2.829 | 17,739 / 11.548 / 2.803 |
| B | 18,733 / 11.046 / 2.816 | 18,475 / 11.113 / 2.789 |

Cells are separate medians, the headline gain is the median of six paired ratios. Three untraced pairs per GPU pair, reversed order between pairs, followed by four separate diagnostic cells. All **16 serving cells pass**, with identical input hash and mean input length. Acceptance estimation, confidence ranking and drafter k=7 remain unchanged on frozen source `888ec6bb`.

The correction changes relative prices. At R=256, `C(825)/C(0)` is **1.022 / 1.157** for the two native startup profiles versus **1.178** for the measured mean curve. All **287/287 priced steady oracle choices** change relative to the native argmax using the same estimated acceptance prefix, with median budget change **−0.255 verified drafts/request**. Changed decisions alone did not produce a serving gain. [Prices, normalized prices and paired gains](results/cost-c256-tp2-analysis-20261019/costs-and-gains.svg).

Coverage is **287/966 steady choices (29.71%)**. Mixed/prefill steps retain native prices for 618 choices (63.98%), another 61 choices lack a complete curve. This remains an empirical decode intervention, the full perfect-cost benefit is unresolved. Mean versus median prices also choose different budgets in 87/287 priced choices. [Decision audit](results/cost-c256-tp2-analysis-20261019/decisions.csv), [cost transfer](results/cost-c256-tp2-analysis-20261019/cost-transfer.csv).

Reducing `max-num-seqs` alone did not make c256 fit on one B300. Two 1,024-request pilots completed but had 49/53 preemptions, p50 running 146, p50 waiting 107/108, and maximum model batch 150. TP2 sustains c256 without queueing or preemption. The unchanged logits buffer limits verification to `b=0..1081-R`, so the dense R=254/255/256 curves contain **828/827/826 points**, not 7R+1. All **2,481 points** have at least five samples, **23,461 primary observations**, with direct mean prices and no fitting or interpolation. Targeted final sampling fills low-count points without changing acceptance estimates.

The [lookup](results/cost-c256-tp2-analysis-20261019/cost-lookup-frozen.json) was frozen before performance runs, SHA-256 `a8a0807d446461ff22223b4b237acde8b080c2c1c4464aa79281e010097018ce`. Analysis excludes failed calibration attempts, uses successful-cell time intervals, separates restarted trace counters, and applies the previously validated one-step timing shift. Both TP ranks agree on recorded budgets, per-request allocations and graph modes. One calibration timeout episode was excluded and replaced. Both workers exited 0, all GPUs were released, nothing published.

Representative serving command inside an empty run directory. `CHECKOUT` contains the [experiment patch](av-cost-experiment.patch) and helper modules, `TARGET` is the shared cached Qwen3.5-35B-A3B snapshot `59d61f3ce65a6d9863b86d2e96597125219dc754`, `LOOKUP` is the frozen JSON, and `RUN` is the run directory:

```bash
mkdir -p "$RUN"
jq -n --arg lookup "$LOOKUP" '{mode:"native",phase:"startup",lookup:$lookup}' > "$RUN/control.json"
CG_SIZES=$(printf '%s\n' 1 2 4 $(seq 8 8 248) $(seq 256 16 1024) $(seq 1280 256 4096) 5120 6144 | paste -sd,)
cd "$RUN"
canhazgpu run --gpus 2 --timeout 50m -- env OMP_NUM_THREADS=1 TENSOR_PARALLEL_SIZE=2 PYTHONPATH="$CHECKOUT" VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=1024 VLLM_AV_COST_CURVES= AV_COST_CONTROL="$RUN/control.json" AV_COST_TRACE="$RUN/cost-steps.jsonl" AV_COST_LOOKUP="$LOOKUP" AV_COST_MIN_SAMPLES=5 AV_COST_SEED=444 ~/.venv/bin/python -m vllm.entrypoints.openai.api_server --model "$TARGET" --served-model-name qwen --language-model-only --max-model-len 16384 --tensor-parallel-size 2 --max-num-seqs 256 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.92 --no-enable-prefix-caching --disable-uvicorn-access-log --host 127.0.0.1 --port 8692 --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.5-35B-A3B-DFlash","revision":"52cb554b4995dede3e2e1bdb129cdb1f3529332b","num_speculative_tokens":7,"draft_sample_method":"probabilistic","enable_adaptive_verification":true}' --compilation-config "{\"cudagraph_capture_sizes\":[$CG_SIZES]}"
```

Representative measured-cost cell using the unchanged [AIPerf template](steady-speedbench.yaml), with `EXPERIMENT` pointing to this experiment directory. Native cells set `mode` to `native`:

```bash
jq -n --arg lookup "$LOOKUP" '{mode:"oracle",phase:"r1-oracle-c256",lookup:$lookup,trace:false}' > "$RUN/control.tmp"
mv "$RUN/control.tmp" "$RUN/control.json"
sed -e 's/__PORT__/8692/g' -e 's/__C__/256/' -e 's/__N__/1024/' "$EXPERIMENT/steady-speedbench.yaml" > "$RUN/c256.yaml"
QUICK_ARTIFACTS="$RUN/artifacts-r1-oracle-c256" timeout 1800 ~/aiperf-venv/bin/aiperf profile --config "$RUN/c256.yaml"
```

Canonical combined data: `results/cost-c256-tp2-analysis-20261019/{metrics,pairs,decisions,points-frozen}.parquet`, preserving AIPerf metrics and outputs. Rebuild with `make fetch-c256-tp2-results`, `make fetch-c256-frozen`, then `make analyze-c256-tp2`. The [queue helper](queue_cost_c256.py) records effective controls and ordering. Sixteen focused tests, patch dry-run, syntax checks and runtime audits pass.
