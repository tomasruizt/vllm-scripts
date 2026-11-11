# Top-k/top-p latency reproduction

`benchmark.py` reproduces the Qwen3.6-35B-A3B DSpark sampling workload without loading weights: FP32 contiguous logits, vocabulary 248,320, k=20, p=0.8, T=0.7, batches 1/4/9/32. It reuses `benchmarks/benchmark_topk_topp.py` from the vLLM checkout for Zipf logits and the FlashInfer masking comparison. Gaussian logits provide a second input distribution. These are synthetic logits, not captured model outputs.

Each invocation restores fresh logits. Correctness compares filtered probabilities with PyTorch. FlashInfer CUPTI timings include the input copy and host gaps between launches, so `cupti_median_us` is an API span, not a single-kernel time. Separate PyTorch profiler traces contain 20 actual GPU durations per kernel. `topk_topp_median_us` is the median duration of `_topk_topp_kernel`, excluding the fresh-input copy and host gaps.

```bash
make launch RUN_ID=topk-topp-20261103-r4
make fetch-results RUN_ID=topk-topp-20261103-r4
make analyze RUN_ID=topk-topp-20261103-r4
```

The remote command is `chg run --gpus 1 --timeout 30m --note topk-topp-microbench -- ~/code/vllm/.venv/bin/python benchmark.py --output results`. The vLLM environment needs pandas, cupti-python matching its CUDA version, and its `bin/` directory on PATH for Ninja. Every attempt uses a new run ID.

Nov 3 reproduction on B300, vLLM `0eb619dc52` (main on 2026-11-03), run `topk-topp-20261103-r4`:

| Input | Rows | Native kernel median |
|---|---:|---:|
| Selected original model call | 4 | 168.866 µs |
| Original trace, all 578 calls | 4 or 9 | 171.521 µs |
| Gaussian / 0.7 | 4 | 170.049 µs |
| Gaussian / 0.7 | 9 | 175.072 µs |
| Zipf / 0.7 | 4 | 195.089 µs |

All 16 native/FlashInfer cases matched PyTorch (max total variation 0). FlashInfer's dense masking path was slower, around 274 µs summed GPU kernel time at batch 4 with Gaussian inputs, excluding the input copy. This is not the fused sampling API, which has a different output contract. Kernel durations reproduce the original slowdown independently of the serving pipeline. The native kernel launches one CTA per row, at most four SMs for the selected call on a 148-SM GPU, and serially scans 31 vocabulary tiles per full pass. Splitting selection across the vocabulary is a candidate optimization, not a measured speedup here.

Raw metrics, profiler traces, source/version metadata and Parquet outputs live in ignored `results/<RUN_ID>/`. Attempts before r4 had dependency failures or CUDA-event fallback and are retained separately.

## Compact selection comparison

`pytorch-compact` uses `torch.topk()`, `flashinfer-compact` uses `flashinfer.top_k()`. Both select 20 sorted values and indices, then run the same PyTorch softmax, cumulative top-p mask and dense scatter. This isolates selector choice while keeping output reconstruction identical. They are prototypes for k=20. Tied logits can change the selected distribution, so they are not production replacements without further work on tie semantics.

```bash
make launch RUN_ID=topk-topp-compact-20261103 BENCH_ARGS='--batches 4 9 192 256 280 --implementations native pytorch-compact flashinfer-compact --repeats 100'
make fetch-results RUN_ID=topk-topp-compact-20261103
make analyze-compact RUN_ID=topk-topp-compact-20261103
make launch RUN_ID=topk-topp-compact-checks-20261103 BENCH_ARGS='--check-only --implementations native pytorch-compact flashinfer-compact'
make fetch-results RUN_ID=topk-topp-compact-checks-20261103
```

`eager_median_us` uses CUDA events without profiling callbacks and includes fresh-input copying, allocation, all kernels and gaps between launches. `gpu_busy_mean_us` sums GPU kernel durations over 20 profiled calls, excluding the input memcpy and idle gaps. These answer different questions. CUPTI spans are retained as diagnostics because its callbacks can inflate eager launch gaps. Edge checks cover BF16-rounded logits, sparse logits, top-k ties, a top-p boundary and all-masked inputs.

The measured comparison and correctness limits are in [the local summary](report/topk-topp-compact-20261103/summary.md). FlashInfer reduces GPU work versus PyTorch at every measured shape, but eager launch overhead limits the low-row-count improvement. Neither compact prototype exactly matches the current kernel on all tie/boundary cases. No production sampler was changed.

## Comprehensive validation

`validate.py` compares exact native compatibility separately from the full-sort reference, retained-logit integrity, finite-row survival and ranking. Its generic prototype accepts uniform k<=128, supports per-row p, and falls back for heterogeneous/disabled/large k and FlashInfer inputs with unaligned row starts. The matrix covers FP32 inputs, BF16-rounded values, sparse/all-masked rows, ties, near ties, large offsets and three memory layouts. Exact-native differences exit with status 2 and remain in the results.

`flashinfer-deterministic` uses `flashinfer.top_k(..., sorted=True, deterministic=True, tie_break=1)` followed by the same compact PyTorch top-p and dense scatter. It matches the captured Qwen sample, but synthetic cases still differ from native. See [the validation report](report/topk-topp-validation-20261103/summary.md) for the full outcome.

```bash
make launch RUN_ID=validation-small SCRIPT=validate.py BENCH_ARGS='--phase small --backends flashinfer-deterministic'
make launch RUN_ID=validation-large SCRIPT=validate.py BENCH_ARGS='--phase large --backends flashinfer-deterministic'
make launch RUN_ID=upstream-tests SCRIPT=existing_tests.py
make launch-capture RUN_ID=real-logits CAPTURE_SCRIPT=capture.py
make launch-capture RUN_ID=serving-ab CAPTURE_SCRIPT=serving.py
make fetch-results RUN_ID=serving-ab
make analyze-serving RUN_ID=serving-ab
```

Capture and serving reuse the pinned sampling config from `specdec_sol/results/qwen36-dspark-av-main-sampling-20261103/`. Each uses one persistent server with prefix caching disabled. `serving.py` compares six seeded coding prompts and three interleaved 30-second measurements per method at c1/c32. Its worker hook is intentionally restricted to this uniform k=20 workload. Upstream tests run unchanged through an adapter that reads k back to the CPU solely to choose the test route. That readback is excluded from timing. SciPy is required by the upstream distribution tests.

`verify_integration.py` exercises the real rejection kernel with 50,000 unique seeds per capture. It compresses only shared zero-probability vocabulary entries, so it is a functional verification test, not full-vocabulary timing or a model-quality evaluation. `explain.py` retains counterexamples. `alignment_repro.py` isolates the FlashInfer crash, with `--graph-safe` selecting the mitigation. The latter verifies one eager call, not CUDA graph capture.

`make analyze-validation` rebuilds the Nov 3 canonical campaign into long-format Parquet under `results/topk-topp-validation-20261103/`. Earlier failed/pilot runs remain separate. Run IDs, exact commands, source snapshots, raw JSONL and profiler traces are retained under ignored run folders.

## Existing vLLM benchmark

The eligible scenario sweep uses the existing benchmark CLI with a custom fixture factory, preserving its timing and correctness checks. It covers uniform top-k K20/K100, heterogeneous top-k [20,100], uniform combined K20/p0.8 and K100/p0.95, heterogeneous combined k=[20,100]/p=[0.8,0.95], and mixed k=[20,100]/p=[1,0.8]. Batch sizes are 1/4/16/64/128/256/512 and vocabularies 65536/131072/151936/248320. Omitting heterogeneous duplicates at B1 leaves 184 cases, three passes and 1,104 timing cells. Host min/max k and Triton's production dispatch hint are prepared outside timing.

```bash
make launch RUN_ID=topk-topp-scenarios-b300-20261104 SCRIPT=sweep_scenarios.py BENCHMARK_SOURCE=/path/to/vllm/benchmarks/benchmark_topk_topp.py GPU_ARGS='--gpu-ids 0 --nonblock'
make fetch-results analyze-scenarios RUN_ID=topk-topp-scenarios-b300-20261104
```

The current variant is named `flashinfer-fastpath`. It has no dense fallback, unsupported inputs raise `NotImplementedError`. Select `--scenarios topk_whole topk_topp_whole` for supported default configurations. Historical runs and their analysis SQL below retain the old `flashinfer-mask` label and archived source snapshots.

The `eligible` preset runs only the 12 eligible original cases: B=4/64/512, V=131,072/248,320 and the two whole-batch top-k scenarios. It uses three alternating-order passes. Eligibility matches FlashInfer's host selection bound<=256 and V>=65,536. The current interface requires CPU `top_k_min` and `top_k_max` bounds, equal bounds skip per-row masking and disabled top-k rows remain unsupported. The benchmark measures every eligible case, a separate profitability policy can be derived from these results later. `analyze-reduced` accepts both the historical and current variant labels and preserves historical summaries.

```bash
make launch-sweep RUN_ID=topk-topp-fastpath-eligible-b300-20261104 SWEEP_TIMEOUT=15m BENCH_ARGS='--preset eligible --implementations triton flashinfer-fastpath --iterations 30'
make fetch-results analyze-reduced RUN_ID=topk-topp-fastpath-eligible-b300-20261104
```

The Nov 4 run uses the modified `vllm/benchmarks/benchmark_topk_topp.py` directly. Its `flashinfer-mask` variant selects compact candidates for uniform k<=128 and retains dense FlashInfer for other scenarios. `--top-k` and `--top-p` override active parameters without changing disabled rows. The benchmark resolves uniform k outside timing. Odd vocabulary sizes request FlashInfer's graph-safe path.

```bash
make launch-existing RUN_ID=vllm-topk-topp-compact-20261104
make fetch-results RUN_ID=vllm-topk-topp-compact-20261104
make analyze-existing RUN_ID=vllm-topk-topp-compact-20261104
```

`run_existing.sh` runs three dense passes with rotated implementation order, one sparse-13 pass and one pass with the original scenario defaults. All calls execute the existing benchmark, with 200 timed iterations per cell except the 20-iteration default check. Timing is the benchmark's mean CUPTI API span, including fresh-input cloning, with cold L2 and no CUDA graphs. These spans include launch gaps and are not individual kernel durations. Outputs include per-cell reference checks, source hash, packages, GPU IDs and commands. `existing.sql` consolidates all scenarios into long-format `metrics.parquet` and `summary.csv`. [Results](report/vllm-topk-topp-compact-20261104/summary.md).

For a single custom invocation, pass the existing benchmark's options through `BENCH_ARGS` to `launch-existing`, using a fresh run ID. This path also preserves the source snapshot and environment metadata.

The expanded B300/H100 campaign uses `sweep_existing.py` to invoke that same entry point, with 5 warmups and 30 timed samples per implementation. It varies batch, vocabulary, k/p, distribution, sparsity and mixed scenarios. The matrix is factorized, with two reversed-order passes for coverage and the k/p cross, and one pass for distribution and odd-vocabulary cases. `make fetch-expanded analyze-expanded` rebuilds the canonical 17,616 implementation cells, excluding superseded partial runs. The H100 resume contributes cases 168 onward. `validate_existing.py` checks the compact path against an independent FP64 prefix-mass oracle. `explain_existing.py` replays Gaussian disagreements, while `diagnose_dense.py` isolates the existing dense fallback's p=1 failure. [Expanded results](report/topk-topp-expanded-20261104/summary.md).

The reduced preset runs B=4/64/512, V=32,768/131,072/248,320 and all six scenarios on a single GPU. It retains the original scenario defaults (k=50 for top-k-only, k=100/p=0.9 for combined filtering, p=0.9 for top-p-only, and the original mixed settings). Two reversed-order passes compare Triton and FlashInfer, with PyTorch used only for correctness checks. `reduced.sql` exports every cell and paired summary, including reference differences and regressions. Negative delta means FlashInfer is faster.

```bash
make launch-sweep RUN_ID=topk-topp-reduced-b300-20261104 SWEEP_TIMEOUT=15m BENCH_ARGS='--preset reduced --implementations triton flashinfer-mask --iterations 30'
make fetch-results RUN_ID=topk-topp-reduced-b300-20261104
make analyze-reduced RUN_ID=topk-topp-reduced-b300-20261104
```

The corrected rerun is `topk-topp-reduced-prod-hint-b300-20261104`, pinned to the same GPU 5. It resolves Triton's production `has_topp_only_rows` hint outside timing. Earlier runs used the conservative default, adding seven unnecessary launches to small uniform top-k+top-p cases. `verify_dispatch.py` validates exact output compatibility and profiles 7→0 launches for uniform combined filtering, with 7→7 retained for mixed batches. [Corrected 54-case results](report/topk-topp-reduced-prod-hint-b300-20261104/summary.md) supersede the earlier reduced comparison for production dispatch.
