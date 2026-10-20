# Task: per-call overhead of the flashinfer TRT-LLM MoE wrapper

Read `README.md` in this folder first.

## Context

In the eager prefill step (~90 ms CPU for ~19 ms GPU), py-spy attributes ~39% of CPU time to the MoE layer. Within it, the flashinfer `trtllm_bf16_moe` path costs ~7% in the C++ launch and ~3% in a per-call autotuner tactic lookup, and a torch-profiler trace shows ~0.1 ms of Python right before 10 `cudaDeviceGetAttribute` calls on every MoE call. With 40 MoE layers per step that adds up. Data: `../results/prefill-step-profile/` (py-spy stacks and torch traces), `../analyze_pyspy.py`. The MoE backend in use is `TrtLlmBf16ExpertsMonolithic` (vLLM) calling flashinfer `trtllm_bf16_moe`.

## What to do

1. Find the Python call path from vLLM's MoE layer into flashinfer `trtllm_bf16_moe` and identify what runs per call: autotuner/tactic lookup, device attribute queries, argument packing, tensor allocations.
2. Microbenchmark the per-call CPU overhead in isolation: call the MoE op in a loop with fixed realistic shapes (e.g. 1,080 and 2,048 tokens, the model's hidden size, experts and top-k) and measure CPU time per call with and without the GPU work (e.g. `torch.cuda.synchronize` only at the end, compare against a pure kernel time from nsys).
3. Prototype caching (tactic per shape, device attributes once per device) in a copy of the vLLM checkout or by monkeypatching flashinfer in a sitecustomize, and measure:
   - per-call CPU overhead
   - solo 1,080-token prefill step time (`make profile-prefill`)
   - a short c128 run with default capture sizes, AV off

## Deliverable

- Breakdown of per-call CPU overhead by component, with file and line references in vLLM and flashinfer.
- Before and after numbers for the microbenchmark, the prefill step and c128 throughput.
- A recommendation: fix in flashinfer, in vLLM's wrapper, or not worth it.
