[Created using AI]

Ran it: 45k-token prompts, TP2, `temperature=1.0`, no collapse. Same for your exact commit.

| Cell (`temperature=1.0`, `top_p=1.0`, k=7) | TP1 | TP2 |
|---|---:|---:|
| 45k prompts, c=4, `main` dbc36f00d | 2.45 | **2.97** |
| 45k prompts, c=4, `1648fefab` (your nightly) | 2.72 | **2.87** |
| 1k prompts, c=32, `main` | 2.38 | 2.31 |
| 1k prompts, fixed seed 42, `main` | 2.27 | 2.40 |
| 1k prompts, `draft_sample_method=probabilistic`, `main` | 2.62 | 2.51 |

Greedy controls on the same host and prompts: 45k TP1 3.12 / TP2 3.15, 1k TP1 2.71 / TP2 2.73 / TP4 2.66. So T=1.0 does cost acceptance here (2.45-2.97 vs 3.12), it just does not interact with TP.

For the `1648fefab` rows I added a git worktree at that commit and reused the precompiled extensions, `vllm.__file__` and `git log` confirm it. `enable_batch_sharded_sampling` and `use_local_argmax_reduction` are both off, as in your command.

Two observations that might narrow it down:

- Your TP2 per-position vector at T=1.0 is exactly 0.000 from position 2 on, while TP2 greedy on the same build is 0.794, 0.662, 0.299, and the draft tokens are the same argmax in both cases (vLLM defaults to `draft_sample_method=greedy`, so temperature does not change what the drafter proposes). That points at the acceptance decision being made against the wrong distribution, not at the drafter degrading.
- The one combination I had not tried before this message is your explicit `cudagraph_mode=FULL_DECODE_ONLY` together with T=1.0. At T=0 the sampler needs no RNG, at T=1.0 it does, and a decode graph that replays RNG state per rank is exactly where a TP-dependent difference could hide. Those cells (`FULL_DECODE_ONLY` and `--enforce-eager`, both at T=1.0, 45k, TP1 and TP2) are running now and I will report them.

Two questions for you:

- Which GPU is this? My TP1 run needs 178 GB of weights, so this has to be B200 or B300, and it would be good to know which.
- Can you try `--enforce-eager` at TP2 with T=1.0 on your side? If acceptance recovers, the graph plus RNG path is where to look, and that would also explain why both v0.33.0 and the nightly collapse at T=1.0 while greedy is fine on the nightly.
