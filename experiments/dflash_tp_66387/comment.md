[Created using AI]

I tried to reproduce this on a B300 node (`main` at `dbc36f00d`, 2026-11-07, `0.33.0rc1.dev1014`) with checkpoint rev `5220c3f` and `num_speculative_tokens: 7`, and I could not. TP2 and TP4 hold the TP1 acceptance length for every workload I tried.

| Cell (mean acceptance length) | TP1 | TP2 | TP4 |
|---|---:|---:|---:|
| 1k-token prompts, c=32 | 2.71 | 2.73 | 2.66 |
| 16k-token prompts, c=8 | 2.79 | 2.76 | - |
| 45k-token prompts, c=4 | 3.12 | 3.15 | - |
| 8k-token prompts, c=64 (preempting) | 2.73 | 2.72 | - |
| 1k-token prompts, `FULL_DECODE_ONLY` | 2.71 | 2.71 | - |

The 45k TP1 cell lands on your TP1 level (3.12 vs 3.12-3.18, per-position 0.756, 0.494, 0.323, 0.216, 0.146, 0.116, 0.073 vs 0.677, 0.465, 0.330, 0.244, 0.183, 0.146, 0.126), so the workload is comparable, and TP2 stays at 3.15 instead of dropping to ~1.2.

Two things that look worth checking on the affected setup:

- The TP4 column is `MiMo-V2.6-Pro-MOPD` with its own bundled drafter, so it is not the same pair as the TP1/TP2 rows.
- Acceptance near 1.0 with intact outputs is also what a mis-loaded drafter or a corrupted draft context looks like. A target-only greedy run at TP1 and TP2 on the same prompts separates those cases. If the target-only text is clean at TP2 while acceptance stays near 1.1, the drafter weights or its context KV are the place to look. The log line naming the loaded drafter, and any fallback warning, would settle it quickly.

Your TP1 run needs ~178 GB on a single device, so B200 or B300. Here target and draft both select `FLASH_ATTN` at TP1 and TP2, and your nightly `1648fefab` is 22 commits behind my build with no spec-decode TP changes in between, so I would not expect a build difference, but confirming the GPU model is cheap.

I did not test the OpenAI server path with `--max-num-seqs 128` and 770 output tokens. The offline cells submit the whole batch at once, which reproduces the scheduling pressure but not the HTTP layer, and acceptance is computed engine side.

<details>
<summary>All measurements, environment and method</summary>

### Environment

- Single B300 node, 275 GiB per GPU, sm_103, 8 GPUs, NVLink.
- vLLM `0.33.0rc1.dev1014+gdbc36f00d`, main on 2026-11-07, precompiled extensions, source checkout on `PYTHONPATH`.
- Target `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` rev `5220c3f`, drafter `<snapshot>/dflash`, `--language-model-only --trust-remote-code`.
- Offline `LLM.generate`, `temperature=0`, `max_tokens=256`, one `do_log_stats()` after generation, which prints the same `SpecDecoding metrics` line the server logs.

### Method

```python
LLM(model="XiaomiMiMo/MiMo-V2.6-Flash-MOPD",
    tensor_parallel_size=tp,
    speculative_config={"method": "dflash", "model": "<snapshot>/dflash",
                        "num_speculative_tokens": 7},
    max_model_len=49152, max_num_seqs=4,
    language_model_only=True, trust_remote_code=True)
```

Prompts come from SPEED-Bench `throughput_1k`, which is natural long-form text. The 45k-token cells join 44 successive prompts into one prompt, so the contexts are distinct text instead of one repeated document. The 8k/c=64 cell pushes 64 requests of 8k tokens through 2 GPUs, so it preempts.

### Per-cell numbers

| Cell | TP | Mean acceptance length | Per-position acceptance rate |
|---|---:|---:|---|
| 1k prompts, c=32 | 1 | 2.71 | 0.658, 0.395, 0.249, 0.163, 0.113, 0.077, 0.051 |
| 1k prompts, c=32 | 2 | 2.73 | 0.674, 0.407, 0.248, 0.162, 0.110, 0.076, 0.048 |
| 1k prompts, c=32 | 4 | 2.66 | 0.657, 0.386, 0.242, 0.151, 0.109, 0.073, 0.045 |
| 16k prompts, c=8 | 1 | 2.79 | 0.693, 0.420, 0.250, 0.168, 0.114, 0.083, 0.067 |
| 16k prompts, c=8 | 2 | 2.76 | 0.683, 0.422, 0.243, 0.168, 0.109, 0.079, 0.056 |
| 45k prompts, c=4 | 1 | 3.12 | 0.756, 0.494, 0.323, 0.216, 0.146, 0.116, 0.073 |
| 45k prompts, c=4 | 2 | 3.15 | 0.759, 0.488, 0.315, 0.207, 0.160, 0.133, 0.093 |
| 8k prompts, c=64 | 1 | 2.73 | 0.663, 0.404, 0.249, 0.162, 0.115, 0.082, 0.056 |
| 8k prompts, c=64 | 2 | 2.72 | 0.664, 0.395, 0.248, 0.161, 0.114, 0.081, 0.056 |
| 1k prompts, `FULL_DECODE_ONLY` | 1 | 2.71 | 0.659, 0.396, 0.249, 0.163, 0.114, 0.078, 0.051 |
| 1k prompts, `FULL_DECODE_ONLY` | 2 | 2.71 | 0.664, 0.402, 0.247, 0.160, 0.112, 0.079, 0.049 |

### Other drafter and target pairs

All TP-invariant, including the two drafter shapes that differ most from the MiMo one.

| Pair | k | TP1 | TP2 | TP4 |
|---|---:|---:|---:|---:|
| `Qwen/Qwen3-4B` + `z-lab/Qwen3-4B-DFlash-b16` (dense, full attention) | 15 | 2.26 | 2.25 | 2.23 |
| same, 16k-token prompts | 15 | 1.59 | 1.62 | - |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` + `-NVFP4-DFlash` (MoE, 2 KV heads, replicated at TP4) | 7 | 3.28 | 3.06 | 3.22 |
| `Qwen/Qwen3.5-4B` + `z-lab/Qwen3.5-4B-DFlash` (mixed sliding/full drafter, hybrid linear-attention target) | 7 | 3.64 | 3.64 | 3.62 |

The Nemotron pair covers a MoE target with a non-causal drafter whose KV heads are replicated at TP4. The Qwen3.5 pair covers a drafter with its own sliding-window KV groups on a target with linear-attention state, which is the closest structural analogue to the MiMo drafter among the small checkpoints. Qwen3.5-4B per-position at TP1 is 0.776, 0.565, 0.400, 0.308, 0.239, 0.195, 0.153 and at TP4 0.786, 0.556, 0.399, 0.303, 0.235, 0.192, 0.151.

</details>
