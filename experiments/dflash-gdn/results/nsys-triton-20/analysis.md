# Nsight Systems: Triton Mamba gather

20-token greedy DFlash request after one warmup, Qwen3.5-4B, FP8, TP1, MRV2, max-model-len 4096, max-num-seqs 8. Capture contains prefill plus four steady verification steps. Original and updated vLLM returned identical response text.

| Recorded across request | Original vLLM | Triton vLLM | SGLang |
| --- | ---: | ---: | ---: |
| Eager kernel executions | 2,078 | 1,358 | 840 |
| Memory copies | 931 | 931 | 143 |
| CUDA graph envelopes (ms) | 24.71 | 24.68 | 24.41 |
| GPU activity union (ms) | 32.73 | 31.78 | 27.60 |
| Gaps between GPU activity (ms) | 68.52 | 67.74 | 32.54 |

The fused gather executes 120 times: 24 GDN groups across five target passes. Replacing seven kernels with one removes exactly 720 launches. CUDA graph execution time is essentially unchanged.

The updated capture still contains four bursts of 24 one-byte host-to-device copies, matching per-group speculative mask construction. Each window from its first to last mask copy contains 185 cudaMemcpyAsync calls; these windows omit the edges of the metadata phase. Thus the remaining copy overhead survived the gather optimization. The original PyTorch trace attributed 192 copies per verification step to GDN builders.

Next candidate: a uniform speculative-verification metadata path that avoids CPU mask transfers and boolean-indexing temporaries, and writes directly into persistent graph buffers.

These are profiled request timelines, not unprofiled ITL. Graph envelopes are not a measure of SM utilization. Historical captures used PyTorch 2.11; the new capture uses 2.13 and CUDA-profiler-API capture with OS runtime tracing. Use operation counts to confirm the mechanism and the unprofiled benchmarks to measure speedup.

Files: [Nsight report](20_tokens.nsys-rep), [SQLite export](20_tokens.sqlite), [analysis data](analysis.json), [server log](server.log), [capture response](capture_response.json), and [capture script](run_profile.py).
