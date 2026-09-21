# SGLang DFlash: 95% memory, BF16 GDN states, prefix caching disabled

One H100 per model, TP=1, FP8 weights, SGLang 0.5.17, the same pinned checkpoints and AIPerf GSM8K workload as the earlier sweep. Output limit 256, EOS enabled. DFlash block size remains 16 (15 speculative tokens plus the current token). ReplaySSM is disabled.

Flags: `--mem-fraction-static 0.95 --mamba-ssm-dtype bfloat16 --disable-radix-cache --chunked-prefill-size 2048`. Server request cap: 32 for 4B, 16 for 27B. Each server starts once and serves all four concurrency levels in ascending order.

## 4B

| Concurrency | Output tok/s | ITL p99 (ms) | AL | Observed max active reqs | Benchmark wall (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 879.9 | 10.13 | 5.628 | 1 | 47.20 |
| 8 | 3,585.8 | 30.37 | 5.570 | 9 | 27.95 |
| 16 | 4,551.7 | 49.02 | 5.462 | 17 | 35.45 |
| 32 | 6,935.9 | 41.45 | 5.498 | 32 | 41.73 |

Server startup: 43.02 s, paid once.

## 27B

| Concurrency | Output tok/s | ITL p99 (ms) | AL | Observed max active reqs | Benchmark wall (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 396.2 | 21.71 | 7.648 | 1 | 87.46 |
| 8 | 1,719.7 | 67.49 | 7.710 | 8 | 41.89 |
| 16 | 2,662.4 | 77.62 | 7.778 | 16 | 49.37 |
| 32 | 2,661.9 | 77.41 | 7.704 | 16 | 83.84 |

Server startup: 57.02 s, paid once. Concurrency 32 queues behind the 16-request server cap.

## Notes and logs

- Requests/warmups: c=1 uses 100/10; c=8 uses 160/16; c=16 uses 320/32; c=32 uses 640/64. Benchmark wall includes warmup and AIPerf overhead; reported throughput excludes them.
- Active counts are observed maxima from `sglang:num_running_reqs`; the table preserves the transient values above client concurrency. AL is the mean sampled SGLang acceptance-length gauge.
- Startup failed with the default 48-request cap and again at 32 with the original 8,192-token prefill chunk. 4B ran out of memory during warmup; 27B could not allocate its cache. Reducing prefill to 2,048 and the 27B cap to 16 produced these successful runs. Failed logs remain in sibling directories `sglang-no-prefix-95-bf16` and `sglang-no-prefix-95-bf16-max32`.
- These results change memory allocation, state precision, prefix caching, prefill size, and server request caps relative to the earlier sweep. Earlier vLLM results have not been rerun with prefix caching disabled; the benchmark launcher now disables it for both engines.
- Shared server logs: `4B/sglang_dflash/server.log` and `27B/sglang_dflash/server.log`. Per-concurrency configuration, benchmark logs, metrics and summaries: `<model>/sglang_dflash/c<concurrency>/`. The repeated startup field in each summary refers to the same single server startup.
