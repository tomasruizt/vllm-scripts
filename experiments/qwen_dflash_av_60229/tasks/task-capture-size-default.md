# Task: what should the default CUDA graph capture list be?

Read `README.md` in this folder first.

## Context

vLLM captures CUDA graphs up to `max_cudagraph_capture_size = min(max_num_seqs × (1 + k) × 2, 1024 on data-center Blackwell, else 512)` (`_set_cudagraph_sizes` in `vllm/config/vllm.py`). The list is 1, 2, 4, multiples of 8 up to 256, then multiples of 16 up to the cap. PR #49390 raised the Blackwell cap from 512 to 1,024 based on one DeepSeek-V4-Flash workload; 2K, 4K or 8K were never measured.

On Qwen3.5-35B-A3B every step with a prefill exceeds 1,024 tokens and runs eagerly. Adding 14 sparse sizes from 1,280 to 6,144 (`cg_sizes` in `../run-steady.sh`) gave +25 to 37% throughput at c32 to c128, for +2.8 GiB of graph memory (−1.9% KV cache) and +50 s of startup. A dense model with much more GPU work per token may gain little and still pay the memory and startup cost.

## What to do

1. Quantify cost and benefit of capture lists on two models: Qwen3.5-35B-A3B (hybrid MoE, ~3B active) and a dense model that fits one B300, e.g. `Qwen/Qwen3.8-27B` (check the shared model mount first, download per AGENTS.md if needed). Variants:
   - default (cap 1,024)
   - default plus sparse sizes up to 2,048, 4,096, 8,192 and `max_num_batched_tokens`
   - default plus dense stride-16 sizes up to 4,096 (to show the cost of naive extension)
2. Per variant, record from the server log: capture time, graph memory, KV cache tokens, engine startup time. Then run AV off at c8, c32, c64 and c128 with the steady harness (`../run-steady.sh` with a `*cg`-style mode or `--compilation-config '{"cudagraph_capture_sizes": [...]}'`), 1 repeat is enough for a first pass.
3. Measure the eager CPU vs GPU time of a prefill step for both models (torch profiler, `make profile-prefill` as a template, or `vllm bench forward` for the GPU side) to estimate where steps become GPU-bound.

## Deliverable

- Table per model and variant: startup, graph memory, KV tokens, throughput per c.
- Where the benefit saturates, and whether any model loses throughput.
- A proposed default rule (e.g. extend the cap to the largest mixed step the scheduler can produce, with sparse geometric spacing, only when eager steps are CPU-bound), with the evidence for it.
