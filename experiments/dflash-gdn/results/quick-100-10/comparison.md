# Qwen3.5-4B: 100-request comparison

Concurrency 1; 10 warmup and 100 measured requests; AIPerf GSM8K; maximum 256 output tokens; EOS enabled; FP8 target weights; TP1; one H100 per run. Original Mamba helper remains active. HTTP access logs disabled.

| Engine | Mode | ITL p50 (ms) | Output tok/s | AL |
|---|---|---:|---:|---:|
| vllm | baseline | 3.18 | 301.6 | — |
| vllm | dflash | 8.74 | 565.6 | 5.65 |
| sglang | baseline | 3.25 | 299.8 | — |
| sglang | dflash | 5.99 | 847.6 | 5.71 |

| Engine | Mode | Server startup (s) | Benchmark total (s) | Warmup (s) | Measured requests (s) |
|---|---|---:|---:|---:|---:|
| vllm | baseline | 50.0 | 109.3 | 8.8 | 84.9 |
| vllm | dflash | 46.0 | 66.1 | 5.2 | 45.3 |
| sglang | baseline | 39.0 | 109.4 | 8.6 | 85.4 |
| sglang | dflash | 41.0 | 49.2 | 3.7 | 30.2 |

ITL is the streamed-chunk interval. Startup is timed from server launch until the health check succeeds, including compilation and graph capture; checks poll every two seconds. Benchmark total includes AIPerf setup, warmup, measurement, and export. Startup depends on compilation cache state.

AL includes the bonus token. vLLM uses measured-phase counter deltas; SGLang uses the mean sampled AL gauge, with different weighting.

Environments: vLLM uses PyTorch 2.13.0+cu130; SGLang uses PyTorch 2.11.0+cu130.
