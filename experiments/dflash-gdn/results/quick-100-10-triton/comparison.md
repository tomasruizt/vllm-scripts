# Triton Mamba block-table gather: before and after

100 measured requests, 10 warmups, concurrency 1, maximum 256 output tokens, EOS enabled; same AIPerf GSM8K workload and engine settings. Both vLLM runs now use the Triton implementation.

| Mode | ITL before (ms) | ITL after (ms) | Tok/s before | Tok/s after | AL before | AL after |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 3.18 | 3.25 | 301.6 | 298.2 | — | — |
| dflash | 8.74 | 8.19 | 565.6 | 608.6 | 5.6523 | 5.6523 |

| Mode | Startup (s) | Benchmark total (s) | Warmup (s) | Measured requests (s) |
|---|---:|---:|---:|---:|
| baseline | 46.0 | 110.4 | 8.6 | 85.8 |
| dflash | 46.0 | 61.6 | 4.7 | 42.1 |

ITL is the median streamed-chunk interval. Startup includes model loading, compilation, and graph capture; readiness is polled every two seconds. Benchmark total includes AIPerf setup, warmup, measured requests, and export. Results are one run per configuration on separate H100 GPUs.

SGLang reference: baseline 3.25 ms ITL / 299.8 tok/s; DFlash 5.99 ms ITL / 847.6 tok/s. The Triton change closes approximately 20% of the observed DFlash ITL gap and 15% of the throughput gap in these runs.
