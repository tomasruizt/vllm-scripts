[Created using AI]

Follow-up on the graph question, same 45k prompts, c=4, `temperature=1.0`, k=7:

| Variant | TP1 | TP2 |
|---|---:|---:|
| `main` dbc36f00d, default cudagraph mode | 2.45 | 2.97 |
| `cudagraph_mode=FULL_DECODE_ONLY` | 2.45 | 2.97 |
| `--enforce-eager` | 2.37 | 2.77 |
| `1648fefab` (your nightly), default mode | 2.72 | 2.87 |

`FULL_DECODE_ONLY` is identical to the default mode here, and eager changes nothing either. Combined with the earlier T=1.0 cells (1k prompts, fixed seed, `draft_sample_method=probabilistic`), acceptance is TP-invariant in every configuration I can build on this host. Both TP1 and TP2 also produce coherent text at T=1.0, I decoded the token streams to check.

One correction on my side: B200 cannot be your GPU. A B200 reports 183 GB, and at `--gpu-memory-utilization 0.95` that leaves 174 GB, below the 178 GB of weights your TP1 run needs. So you are on B300 like me, and the GPU class is not the difference.

That leaves the serve path (4 TP2 replicas over HTTP) and your prompt set. Two things that would narrow it quickly:

- At TP2 with T=1.0, run one cell with "draft_sample_method": "probabilistic" in `--speculative-config`. With greedy drafts, T=0 verifies by token equality while T>0 uses the ratio test against a one-hot proposal, so that single switch is the only acceptance-side difference between your working and broken configs. If probabilistic drafts at T=1.0 are healthy at TP2, the bug is in the greedy one-hot ratio branch rather than in the drafter or its context KV. On this host both draft methods are TP-invariant at T=1.0.
- Your `--enforce-eager` result at TP2 with T=1.0, and one full engine log from a collapsing run. The per-engine `SpecDecoding metrics` lines plus any drafter warnings would tell us whether all ranks are equally bad or whether the mean is smeared across engines.

If it helps, I can run your exact request payload (one prompt plus the sampling parameters) verbatim here, which would separate the prompt set from everything else.
