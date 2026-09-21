# 27B DFlash cache-group size comparison

Qwen3.5-27B with z-lab/Qwen3.5-27B-DFlash, vLLM MRV2 with Triton Mamba gather, FP8 weights, TP1, concurrency 1, 100 measured requests plus 10 warmup, AIPerf spec_al_gsm8k, max_completion_tokens=256, ignore_eos unset. Same server settings as the new 4B group-size sweep: max-model-len 32768, max-num-seqs 128. Four runs used separate H100 GPUs through canhazgpu run; all allocations released on completion.

| Group size | ITL p50 (ms) | Output tok/s | AL | Reported KV capacity (tokens) | Server startup (s) | Benchmark wall clock (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 19.649 | 330.3 | 7.406799 | 215,934 | 158.0 | 102.7 |
| 2 | 18.908 | 348.7 | 7.406799 | 207,649 | 160.0 | 98.1 |
| 4 | 18.585 | 354.8 | 7.406799 | 192,917 | 164.1 | 96.7 |
| 8 | 18.430 | 361.0 | 7.406799 | 177,708 | 160.0 | 95.5 |
| 16 | 18.165 | 367.9 | 7.406799 | 140,773 | 152.0 | 97.9 |

ITL is streamed inter-chunk latency, not TPOT. Benchmark wall clock includes warmup and AIPerf setup/export. Startup includes fresh compilation.

Group size 16 was run separately after the four-way sweep, using the same benchmark settings and an automatically released canhazgpu allocation. Versus size 8, throughput increased 1.9%, ITL fell 1.4%, and reported cache capacity fell 20.8%. Acceptance length was unchanged. This small speed difference needs repeated runs to establish its size reliably.

Group size 8 improves throughput by 9.3% and reduces ITL by 6.2% versus size 1, with 17.7% lower reported cache capacity. Acceptance length is identical across all four runs. These are single runs on different GPUs with overlapping host activity; identical AL is not a full output-correctness evaluation.

Historical 27B SGLang DFlash throughput was 364.5 tok/s, but that run used 200 measured requests, 20 warmup, the older custom GSM8K prompt format, and different server limits. It is context only, not a matched comparison to this sweep.

Each numbered directory contains vllm_dflash/server.log, benchmark.log, run_config.json, summary.json, metrics snapshots and AIPerf exports. run_config.json records the exact commands and pinned model revisions. Reproduce using benchmarks/dflash_4b/run.py with --model-size 27B and --attn-group-size 1/2/4/8.
