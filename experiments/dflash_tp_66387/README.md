# DFlash acceptance at TP > 1 (vLLM issue #66387)

Trying to reproduce the reported acceptance collapse at `--tensor-parallel-size > 1` with the exact model from the issue, and with smaller proxy pairs.

## Setup

- Host: one B300 node (275 GiB per GPU, sm_103), 8 GPUs.
- vLLM: `0.33.0rc1.dev1014+gdbc36f00d` (main on 2026-11-07), used through `PYTHONPATH`, precompiled extensions.
- The reporter used `v0.33.0` and `0.33.0rc1.dev992+g1648fefab` (2026-11-07). `1648fefab` is an ancestor of the build above, 22 commits back, and none of those commits touch spec decode except a DSpark Gumbel-sampling fusion.
- Workload: offline `LLM.generate`, greedy, fixed prompt set (SPEED-Bench `throughput_1k` prompts), spec-decode stats from one `do_log_stats()` at the end. Long-context cells join successive prompts (`--concat-prompts`) instead of repeating one.
- Measurement: `SpecDecoding metrics: Mean acceptance length` and the per-position vector, the same numbers the issue quotes.

Example cell (45k-token prompts, TP2):

```sh
python run_acceptance.py --tp 2 --prompts throughput_1k.jsonl \
  --model XiaomiMiMo/MiMo-V2.6-Flash-MOPD --draft <snapshot>/dflash \
  --num-spec-tokens 7 --language-model-only --trust-remote-code \
  --concat-prompts 44 --num-prompts 4 --max-model-len 49152 --max-num-seqs 4
```

## Exact reported pair

`XiaomiMiMo/MiMo-V2.6-Flash-MOPD` (rev `5220c3f`) with its bundled `dflash/` drafter, `num_speculative_tokens=7`.

| Cell | TP1 | TP2 | TP4 |
|---|---:|---:|---:|
| 1k prompts, c=32, 256 out | 2.71 | 2.73 | 2.66 |
| 16k prompts, c=8 | 2.79 | 2.76 | - |
| 45k prompts, c=4 | **3.12** | **3.15** | - |
| 8k prompts, c=64 (preempting) | 2.73 | 2.72 | - |
| 1k prompts, `FULL_DECODE_ONLY` | 2.71 | 2.71 | - |

TP1 per-position rates at 45k: 0.756, 0.494, 0.323, 0.216, 0.146, 0.116, 0.073. That reproduces the reported TP1 level (3.12-3.18, 0.677, 0.465, 0.330, 0.244, 0.183, 0.146, 0.126) but TP2/TP4 stay at the TP1 level instead of collapsing to 1.1-1.2.

## Proxy pairs

Every pair is TP-invariant, so the collapse is not a general property of the DFlash TP path.

| Pair | k | TP1 | TP2 | TP4 |
|---|---:|---:|---:|---:|
| `Qwen/Qwen3-4B` + `z-lab/Qwen3-4B-DFlash-b16` | 15 | 2.26 | 2.25 | 2.23 |
| same, 16k context | 15 | 1.59 | 1.62 | - |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` + `-NVFP4-DFlash` | 7 | 3.28 | 3.06 | 3.22 |
| `Qwen/Qwen3.5-4B` + `z-lab/Qwen3.5-4B-DFlash` | 7 | 3.64 | 3.64 | 3.62 |

The Nemotron pair covers MoE plus a 2-KV-head drafter with head replication at TP4. The Qwen3.5 pair covers a mixed sliding/full drafter on a hybrid linear-attention target. Both are non-causal drafters.

## Sampling at temperature 1.0

The reporter's later reply: with greedy sampling the nightly holds TP1's level at TP2, but at the server default (`temperature=1.0` under `--generation-config vllm`) it collapses to 1.17 at TP2, and v0.33.0 collapses at both temperatures. So the remaining claim is a T>0-only TP collapse, 45k prompts, DP4 x TP2.

All cells below are `temperature=1.0`, `top_p=1.0`, k=7, MiMo with its bundled drafter. No cell collapses.

| Cell | TP1 | TP2 |
|---|---:|---:|
| 45k prompts, c=4, `main` dbc36f00d | 2.45 | 2.97 |
| 45k prompts, c=4, `1648fefab` (reporter's nightly) | 2.72 | 2.87 |
| 45k prompts, c=4, `FULL_DECODE_ONLY` | 2.45 | 2.97 |
| 45k prompts, c=4, `--enforce-eager` | 2.37 | 2.77 |
| 1k prompts, c=32 | 2.38 | 2.31 |
| 1k prompts, fixed seed 42, token streams compared | 2.27 | 2.40 |
| 1k prompts, `draft_sample_method=probabilistic` | 2.62 | 2.51 |

Greedy controls: 45k TP1 3.12 / TP2 3.15, 1k TP1 2.71 / TP2 2.73 / TP4 2.66. So T=1.0 costs acceptance here, it just does not interact with TP.

The `1648fefab` cells run from a git worktree at that commit with the newer precompiled extensions copied in (`vllm.__file__` and `git log` verified, `third_party` submodules copied over). Decoding the seed-42 token dumps for TP1 and TP2 at T=1.0 gives coherent English in both, so nothing is broken on this host beyond normal TP numerics.

## Remaining differences from the report

- The reporter cannot be on B200: 183 GB at `--gpu-memory-utilization 0.95` leaves 174 GB, below the 178 GB of weights their TP1 run needs. So B300, same as this host. `FLASH_ATTN` is selected for target and draft at TP1 and TP2 here.
- The TP4 datapoint in the issue is a different model, `MiMo-V2.6-Pro-MOPD`, with its own drafter, so that column is not the same pair as TP1/TP2.
- Untested here: the OpenAI server path with 4 x TP2 replicas, `--max-num-seqs 128` and 770 output tokens. Acceptance is computed engine-side, so the HTTP layer is unlikely to matter, but the replica topology and per-request arrival pattern are the only structural differences left, along with the prompt set.
- Sharpest discriminator for the reporter, since it is one cell: at TP2 with T=1.0, switch `draft_sample_method` to `probabilistic`. With greedy drafts, T=0 verifies by token equality while T>0 uses the ratio test against a one-hot proposal, so that switch isolates the acceptance branch from the drafter and its context KV.
