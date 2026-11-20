# AV within-step allocation headroom

Question: how much would a better per-request draft allocation gain inside one AV step, at the same budget? And does per-request uncertainty (calibration offset, UCB, Thompson) capture any of it?

Method: vLLM branch `tomasruizt/vllm:av-alloc-headroom` (main 89e860de00 from 2026-11-17 plus a trace patch). `AV_TRACE_VERIFY_ALL=1` verifies every draft, so each request's true accepted length is known, while the budget AV would have chosen is still logged. [replay.py](replay.py) replays policies on that trace at that budget. The per-request policies learn only from labels they verified themselves (censored, like production). A second server runs normal AV to check the replay: replayed greedy widths match production in 99.96% of 12,724 steps.

Setup: Qwen3-4B + `RedHatAI/Qwen3-4B-speculator.dflash2` (k=8), 1x B200, temp 0, max 1,024 output tokens, c in {16, 32, 64, 128}, HumanEval / math_reasoning / writing from `RedHatAI/speculator_benchmarks`.

```
vllm serve Qwen/Qwen3-4B --max-model-len 16384 --max-num-seqs 128 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.85 --no-enable-prefix-caching --speculative-config '{"model":"RedHatAI/Qwen3-4B-speculator.dflash2","num_speculative_tokens":8,"method":"dflash","enable_adaptive_verification":true}'
```

## Findings (2026-11-20)

Tokens per step vs greedy, at the same budget (so the same step cost):

| budget | oracle | per-request calib | UCB λ=1 | Thompson |
|---|---|---|---|---|
| AV's own, c32-128 | +0.4 to +1.3% | -0.1 to +0.05% | -0.1 to +0.06% | -0.2 to -0.04% |
| AV's own, c16 | +3.9 to +6.6% | ~0 | ~0 | -0.3% |
| forced 50% of drafts | +5.9 to +14.3% | -0.25 to +0.07% | -0.26 to +0.03% | -0.5 to -0.2% |

- At c>=32 AV already verifies 63-90% of drafts on this setup, so there is little left to reallocate.
- The oracle gap is per-step luck (which draft gets rejected), not a persistent per-request bias. Per-request mean residual (observed minus predicted acceptance) has an sd of only 0.013-0.033, and step-to-step residual correlation is 0.01-0.05. Writing shows a stable per-request offset (split-half correlation ~0.6), but it is too small to change rankings.

### Mixed batches and a 27B target (2026-11-20)

Mixed workload: all 9 `speculator_benchmarks` subsets in one seeded shuffle, thinking off, c in {32, 64, 128}. Qwen3.8-27B + `RedHatAI/Qwen3.8-27B-speculator.dflash2` (k=7) and Qwen3-4B, each on one B200. AV verifies 58-85% of drafts.

| | oracle | hindsight per-request offset | calib | UCB λ=1 | Thompson |
|---|---|---|---|---|---|
| 27B mixed | +1.6 to +3.0% | -0.02 to +0.08% | -0.14 to -0.05% | -0.11 to -0.05% | -0.31 to -0.10% |
| 4B mixed | +1.1 to +1.7% | +0.07 to +0.08% | ~-0.05% | -0.08 to -0.04% | -0.23 to -0.11% |

- Mixing raises the per-request residual sd (4B: 0.04 vs 0.013-0.033), but even the hindsight offset (fitted on the request's whole uncensored history) gains <0.1%. Per-request information is not where the oracle gap is.
- [recalib.py](recalib.py): a per-position 20-bin calibration of the same q_max feature, fitted on c32/c64 and tested on c128, changes tokens by -0.06 to +0.02% while the oracle has +1.1 to +9.7% at the same budgets. The online logistic already extracts what q_max offers. Closing the gap needs new lossless per-draft features.

```
vllm serve Qwen/Qwen3.8-27B --max-model-len 16384 --max-num-seqs 128 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.85 --no-enable-prefix-caching --speculative-config '{"model":"RedHatAI/Qwen3.8-27B-speculator.dflash2","num_speculative_tokens":7,"method":"dflash","enable_adaptive_verification":true}' --default-chat-template-kwargs '{"enable_thinking": false}'
```

### Settings with oracle headroom (2026-11-20)

Two confounders fixed. CUDA graphs now cover every tested size (`max_cudagraph_capture_size = max_num_seqs * (K+1)`, the Blackwell default of 1,024 tokens ran the 4B c128 verify-all step eager). And the 1GB rejection-sampler chunk caps the AV budget at `2**30 / (vocab * 4)` logits (1,766 for Qwen3-4B, 1,081 for Qwen3.8-27B), which censored verify-all at c>=256. The branch lifts it with `AV_TRACE_MAX_CHUNK_BYTES=8GiB`, the replay reapplies it offline as `chunk_cap`. Mixed prompts, thinking on, temp 0.

| model | c (mean running) | budget | drafts verified | oracle vs greedy |
|---|---|---|---|---|
| Qwen3.8-27B TP2 | 256 (128) | AV cost model (cap not binding) | 58% | **+6.6%** |
| Qwen3.8-27B TP2 | 128 (86) | AV cost model | 68% | +3.3% |
| Qwen3-4B | 512 (393) | production chunk cap | 34% | **+8.8%** |
| Qwen3-4B | 512 (393) | AV cost model, cap lifted | 68% | +2.1% |
| Qwen3-4B | 256 (194) | AV cost model | 76% | +0.8% |
| both | 128 to 512 | forced 50% | 50% | +9.2 to +10.7% |

Per-request policies stay at ~0 (hindsight offset +0.1 to +0.2%, UCB -0.2 to 0%) in all of them.

Run: `make launch watch fetch-results analyze` (see the Makefile variables for model, prompts and GPUs).
