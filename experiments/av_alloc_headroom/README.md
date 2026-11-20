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

Run: `make launch watch fetch-results analyze`.
