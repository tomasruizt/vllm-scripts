# Task: why are 1-token decode requests slower than 2-token ones?

Read `README.md` in this folder first.

## Context

AV profiles step costs at startup with dummy batches (`_dummy_run(num_tokens=N)` in `vllm/v1/worker/gpu/model_runner.py`): N tokens spread over `min(N, max_num_seqs=512)` requests, each with a 1,024-token KV context (`set_dummy_context` in `vllm/v1/worker/gpu/input_batch.py`). The profiled verify time rises to 19.4 ms at N=512 (512 requests × 1 query token), then falls linearly to 11.8 ms at N=1,024 (the same 512 requests × 2 query tokens). So a request with 1 query token is profiled as more expensive than one with 2. The likely cause is a slower kernel path for `q = 1` in GDN (linear attention) or full attention, but this is not verified.

This matters twice: AV's cost model is distorted by it, and if real `q = 1` decode (no speculative decoding, or requests with no admitted drafts) takes the slow path, plain decode may be leaving performance on the table.

## What to do

1. Reproduce the dip with exact shapes. Preferred: `vllm bench forward` on a recent main (see README) with cases like `decode=512*1@1024`, `decode=512*2@1024`, `decode=128*1@1024`, `decode=128*2@1024`, `decode=128*8@1024`, with the Qwen3.5 model and the same speculative config. Alternative without bench forward: time `_dummy_run` directly in the PR checkout.
2. For `512*1` vs `512*2`, collect a torch profiler or nsys kernel breakdown and identify which kernels differ (names, counts, time). Check the GDN metadata builder (`vllm/v1/attention/backends/gdn_attn.py`) and the attention backend for branches on query length.
3. Check whether the same slow path is taken in real serving: plain decode without speculative decoding at c128 (query length 1), compared with spec decode.
4. If a cheaper path exists for `q = 1`, estimate the gain from routing `q = 1` through it (e.g. treat it as a 1-token verify), and whether that is correct.

## Deliverable

- Table of GPU time per shape, with the commands used.
- The kernel(s) responsible and the code branch that selects them (file and line).
- Whether real `q = 1` decode is affected, and the estimated gain of a fix.
- A short recommendation: fix in the kernel/backend, in AV's profiling shapes, or both.
