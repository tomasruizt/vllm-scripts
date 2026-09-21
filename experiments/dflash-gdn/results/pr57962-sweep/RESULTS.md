# DFlash benchmarks: main vs PR #57962 vs PR2 #52297

One H100 per run, TP=1, FP8, MRV2, AIPerf GSM8K, up to 256 output tokens with EOS stopping enabled. Concurrency is the number of outstanding client requests. Main and PR use identical server settings, dependencies, model snapshots, and request counts at each concurrency.

## 4B output tokens/s

| Concurrency | Main | PR | PR vs main | PR2 | PR2 vs main | SGLang | active reqs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 523.6 | 761.8 | +45.5% | 717.4 | +37.0% | 879.9 | Same as concurrency |
| 8 | 2,311.6 | 2,502.7 | +8.3% | 2,841.7 | +22.9% | 3,585.8 | SGLang: 9 |
| 16 | 2,116.0 | 2,284.9 | +8.0% | — | — | 4,551.7 | SGLang: 17 |
| 32 | 3,324.5 | 3,564.4 | +7.2% | — | — | 6,935.9 | Same as concurrency |

## 4B ITL p99 (ms)

ITL measures the interval between streamed chunks; each chunk can contain multiple accepted tokens. Values come from `inter_chunk_latency.p99` in each run's `aiperf/profile_export_aiperf.json`. Lower is better.

| Concurrency | Main | PR | PR vs main | PR2 | PR2 vs main | SGLang | active reqs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 12.83 | 9.04 | -29.5% | 8.06 | -37.2% | 10.13 | Same as concurrency |
| 8 | 61.66 | 61.99 | +0.5% | 47.66 | -22.7% | 30.37 | SGLang: 9 |
| 16 | 68.42 | 66.63 | -2.6% | — | — | 49.02 | SGLang: 17 |
| 32 | 80.91 | 74.42 | -8.0% | — | — | 41.45 | Same as concurrency |

## 27B output tokens/s

| Concurrency | Main | PR | PR vs main | PR2 | PR2 vs main | SGLang | active reqs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 313.8 | 340.5 | +8.5% | 336.9 | +7.3% | 396.2 | Same as concurrency |
| 8 | 1,171.8 | 1,263.1 | +7.8% | — | — | 1,719.7 | Same as concurrency |
| 16 | 1,465.7 | 1,449.0 | -1.1% | — | — | 2,662.4 | Main/PR: 14 |
| 32 | 1,460.5 | 1,645.0 | +12.6% | — | — | 2,661.9 | Main/PR: 14; SGLang: 16 |

## 27B ITL p99 (ms)

| Concurrency | Main | PR | PR vs main | PR2 | PR2 vs main | SGLang | active reqs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 25.01 | 20.62 | -17.6% | 20.31 | -18.8% | 21.71 | Same as concurrency |
| 8 | 135.84 | 131.87 | -2.9% | — | — | 67.49 | Same as concurrency |
| 16 | 117.09 | 128.75 | +10.0% | — | — | 77.62 | Main/PR: 14 |
| 32 | 116.21 | 109.47 | -5.8% | — | — | 77.41 | Main/PR: 14; SGLang: 16 |

## Workload and interpretation

PR2 was measured at 4B c=1, c=8, and 27B c=1, selected using the earlier SGLang results. PR2 matched concurrency at all three points; — means not run.

Active reqs are the **observed maximum** during the measured phase, from `vllm:num_requests_running` or `sglang:num_running_reqs` in each run's `aiperf/server_metrics_export.json`.
The column lists only discrepancies from client concurrency; unlisted engines match concurrency.
These are sampled engine-reported gauges rather than configured limits or per-step decode batch sizes; their update timing differs between engines.
SGLang's 4B c=8 and c=16 gauges report maxima of 9 and 17; the table preserves those observed values.

| Concurrency | Measured requests | Warmups |
| ---: | ---: | ---: |
| 1 | 100 | 10 |
| 8 | 160 | 16 |
| 16 | 320 | 32 |
| 32 | 640 | 64 |

- These are single runs on a shared host, with independent benchmarks running in parallel on separate GPUs. Throughput excludes server startup; each run's `summary.json` records startup and benchmark wall time separately.
- PR #57962 metadata reuse applies only to pure speculative-decode batches plus padding; mixed batches retain the existing preparation path. Cache-hit frequency was not instrumented in this sweep. PR2 moves common GDN metadata preparation out of per-group builds, including mixed batches.
- The vLLM reproduction settings use CUDA-graph capture sizes up to 128 tokens. A full speculative batch of 16 or 32 requests can require 256 or 512 tokens, respectively, so graph coverage changes across this sweep. These are results for a fixed configuration, not individually tuned configurations for each concurrency.
- SGLang results now use 95% GPU memory, BF16 GDN states, prefix caching disabled, and a 2,048-token prefill chunk. Server request caps are 32 for 4B and 16 for 27B; 27B could not fit 32. Each SGLang server was reused across concurrency levels. The vLLM runs still use 92% memory, BF16 states, and prefix caching enabled, so engine comparisons include these configuration differences.
- At c=32, SGLang throughput increased from 5,142.9 to 6,935.9 tok/s for 4B (+34.9%) and from 891.0 to 2,661.9 tok/s for 27B (+198.8%) compared with its earlier configuration. These gains combine multiple configuration changes and increased active-request capacity.
- Main and PR acceptance lengths match exactly at c=1: 5.652 for 4B and 7.407 for 27B. Higher-concurrency acceptance lengths vary with batching; all values remain in the individual summaries. SGLang AL is a sampled gauge average, while vLLM AL is weighted by draft iterations.

## Revisions and artifacts

- Main: `8b98b7d0b465920740cfa1acf5dde1b6a57d8d52` (September 21, 2026), the PR's exact base. This is a development checkout of `main`, not a tagged release; the recorded package version is `0.1.dev21702+g8b98b7d0b.precompiled`.
- PR: `3ea30065231a57e5ef3f40ff85b38643d51c8895` ([draft #57962](https://github.com/vllm-project/vllm/pull/57962)).
- PR2: [#52297](https://github.com/vllm-project/vllm/pull/52297), head `a8db1fe32ac19c2296bc0e9faedf9550ee56dd2d`, merged without code conflicts with the same main base `8b98b7d0b4` as local commit `d31939925f5f98ad86604266abb3b2b14a94b663`. Worktree: `vllm-pr52297`, branch: `bench/pr52297`. Its existing GDN metadata tests passed (9 tests); the benchmark uses identical compiled extensions to main. The recorded package version is `0.29.1rc1.dev478+gd31939925.precompiled`; fetching release tags changed version-string generation, while the main source base remains the same.
- SGLang: 0.5.17.
- [Sweep script](run_sweep.sh). vLLM results are under `<model>/<main|pr|pr2>/c<concurrency>/vllm_dflash/`, with `run_config.json`, `summary.json`, server/benchmark logs, and AIPerf exports.
- Current SGLang results: [report and settings](../sglang-no-prefix-95-bf16-prefill2048/RESULTS.md). Artifacts are under `../sglang-no-prefix-95-bf16-prefill2048/<model>/sglang_dflash/c<concurrency>/`, with a shared `server.log` one directory above. These replace the earlier SGLang values in all tables.
