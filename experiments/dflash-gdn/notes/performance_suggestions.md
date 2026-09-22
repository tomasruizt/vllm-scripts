# Suggestions for improving vLLM DFlash performance

## Evidence

The Qwen3.5-4B, 20-token Nsight captures show nearly equal aggregate CUDA graph execution time, but substantially more work outside the graphs in vLLM.

| Measured across the profiled request | vLLM | SGLang |
|---|---:|---:|
| CUDA graph execution time | 24.7 ms | 24.4 ms |
| GPU busy time, including other recorded operations | 32.7 ms | 27.6 ms |
| Gaps between recorded GPU activity | 68.5 ms | 32.5 ms |
| `cudaMemcpyAsync` calls | 931 | 143 |

These measurements include prefill and profiling overhead; they are not steady-state ITL measurements.
Graph-level tracing does not expose individual kernels inside the graphs, and gaps alone do not prove a specific CPU bottleneck.
The captures are preserved in `profiles.tar.gz`, under `profiles/nsys/{vllm,sglang}/20_tokens.sqlite` and the corresponding `.nsys-rep` files.

The unprofiled benchmarks used concurrency 1, 200 measured GSM8K requests after 20 warmups, 256 requested output tokens, FP8 target weights, and one H100 per engine.
ITL here means the median interval between streamed chunks, not TPOT.

| Model | vLLM DFlash ITL | SGLang DFlash ITL | vLLM DFlash tok/s | SGLang DFlash tok/s |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | 8.80 ms | 5.99 ms | 569.9 | 829.2 |
| Qwen3.5-27B | 20.06 ms | 17.93 ms | 318.2 | 364.5 |

SGLang's throughput advantage shrinks from 45.5% on 4B to 14.6% on 27B.
This is consistent with overhead becoming less important as model computation grows, but the drafter and acceptance length also change, so this is not a controlled isolation of host overhead.

## 1. Prepare shared GDN metadata once per batch

This is the first change to implement and measure.
The PyTorch trace contains 24 calls to `GDNAttentionMetadataBuilder.build` per steady decoding step.
Those calls repeat sequence classification, index construction, copies, and updates to persistent graph buffers.
Nsight records 96 one-byte host-to-device copies across the request, consistent with repeated mask preparation across four decoding steps; direct attribution would require additional instrumentation.
Do not use the inflated PyTorch CPU durations as estimates of production savings.

Relevant vLLM code:

- `vllm/v1/worker/gpu/model_states/mamba_hybrid.py`: `MambaHybridModelState.prepare_attn`.
- `vllm/v1/worker/gpu/attn_utils.py`: `build_attn_metadata` loops over attention groups and invokes their builders.
- `vllm/v1/attention/backends/gdn_attn.py`: `GDNAttentionMetadataBuilder.build` constructs speculative masks and indices, then copies values into graph buffers.

Proposed changes:

- Compute sequence classification, query offsets, token indices, and accepted counts once per batch and share them across compatible layers.
- Preallocate invariant indices and masks for uniform 16-token verification batches.
- Update group-specific state indices using a batched GPU kernel, preserving distinct mappings and buffers where required.
- Keep the general builder for mixed prefill/decode batches and unsupported layouts.
- Preserve padding sentinels, rejection handling, and stable tensor addresses required by captured graphs.

SGLang's `MambaAttnBackendBase._replay_metadata` in `sglang/srt/layers/attention/hybrid_linear_attn_backend.py` provides a useful comparison: it updates persistent buffers and has a fused state-index replay path.
Its GDN backend inherits this metadata handling.

## 2. Capture DFlash context preparation for uniform decoding

In `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`, `DFlashSpeculator.propose` explicitly performs context KV preparation eagerly before replaying the draft graph.
The draft forward and sampling are already captured.

Proposed changes:

- Extend graph coverage to context projection, normalization, RoPE, and KV insertion for uniform decoding shapes.
- Use stable input and output buffers, with rejected and padded positions masked on the GPU.
- Retain a separate path for variable prefill shapes.
- Check sliding-window groups, cache eviction, and rejected suffixes before expanding the fast path beyond the measured case.

This is an opportunity identified in vLLM's implementation; it is not a claim that SGLang captures all equivalent context work.

## 3. Fuse remaining operations across groups and layers

vLLM prepares DFlash inputs separately for draft KV groups.
In `vllm/model_executor/models/qwen3_dflash.py`, `precompute_and_store_context_kv` already fuses the context KV projection but still inserts KV values through a per-layer loop.

Proposed changes:

- Batch input preparation across compatible KV groups.
- Fuse context KV writes across layers where cache layouts permit it.
- Reuse buffers to reduce intermediate allocations and small copy launches.

Prioritize these after metadata reuse and measure their incremental benefit.
SGLang also has per-layer context-cache writes in its implementation, so the existence of this loop alone does not explain the performance gap.

## Validation and expected outcome

These changes plausibly address a substantial part of the 2.81 ms unprofiled 4B ITL gap, but matching SGLang is not established and no per-change speedup has been measured.

1. Implement metadata reuse first, keeping each subsequent optimization independently measurable.
2. Rerun the identical unprofiled 200-request workload on both 4B and 27B, with the same input order and GPU configuration.
3. Compare output throughput, streamed-chunk ITL, TTFT, acceptance length, and output correctness.
4. Use short Nsight captures to check whether copies, small kernel launches, and gaps decrease during steady decoding.
5. Validate mixed batches, padding, rejected tokens, and cache-state handling before treating the optimization as general.

The analysis applies to the benchmarked vLLM commit `b6e7c1f1f` and the installed SGLang 0.5.17 sources.
The suggestions are implementation hypotheses supported by the traces and source inspection, not completed fixes.
