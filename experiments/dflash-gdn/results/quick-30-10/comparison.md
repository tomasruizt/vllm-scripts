# Qwen3.5-4B quick vLLM benchmark

GPU 2; concurrency 1; 10 warmups and 30 measured GSM8K requests; maximum 256 output tokens, EOS enabled; MRV2; FP8 target weights. Original Mamba helper remains active.

| Mode | ITL p50 (ms) | Output tok/s | AL | Server startup (s) | Benchmark wall time (s) | Warmup (s) | Measured requests (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 3.15 | 307.1 | — | 44.0 | 48.3 | 8.5 | 25.0 |
| dflash | 9.21 | 578.4 | 6.05 | 114.0 | 34.5 | 6.0 | 13.3 |

ITL is the streamed-chunk interval. Startup includes model loading, compilation, and graph capture. Benchmark wall time includes AIPerf initialization, warmup, measured requests, and export. Wall times are approximate, reconstructed from file creation timestamps. The failed initial baseline startup is excluded; its log is preserved in vllm_baseline-startup-failed/server.log.
