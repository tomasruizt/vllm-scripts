# DFlash cache-group size comparison

Qwen3.5-4B, vLLM MRV2 with Triton Mamba gather, FP8 weights, TP1, concurrency 1, 100 measured requests plus 10 warmup, AIPerf spec_al_gsm8k, max_completion_tokens=256, ignore_eos unset. Four runs used separate H100 GPUs through canhazgpu run; GPUs were released on completion.

| Group size | ITL p50 (ms) | Output tok/s | AL | Reported KV capacity (tokens) | Server startup (s) | Benchmark wall clock (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9.049 | 555.5 | 5.652278 | 838,554 | 116.0 | 67.9 |
| 2 | 6.220 | 785.5 | 5.652278 | 773,369 | 116.0 | 53.1 |
| 4 | 6.032 | 824.1 | 5.652278 | 669,348 | 116.0 | 51.2 |
| 8 | 5.889 | 844.4 | 5.652278 | 575,168 | 116.0 | 50.1 |

ITL is streamed inter-chunk latency, not TPOT. Benchmark wall clock includes warmup and AIPerf setup/export. Startup includes fresh compilation under the new configuration hashes.

Earlier SGLang DFlash reference with the same request settings: 847.6 tok/s, ITL 5.992 ms, AL 5.709 (sampled gauge, different weighting). It was not rerun concurrently. Group size 1 was slower than the previous Triton run (608.6 tok/s); these are single runs on different GPUs with overlapping host activity, so small differences need repetition. Identical AL is not a full output-correctness evaluation.

Each numbered directory contains vllm_dflash/server.log, benchmark.log, run_config.json, summary.json, metrics snapshots and AIPerf exports. run_config.json records the exact commands. The benchmark script accepts --attn-group-size to reproduce each setting.
