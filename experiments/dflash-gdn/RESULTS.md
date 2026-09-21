# DFlash comparison

One H100, TP=1, FP8 weights, GSM8K, up to 256 output tokens, EOS enabled, 15 proposed tokens (SGLang block size 16).

Main/PR/earlier PR2: 92% memory, BF16 GDN states, prefix caching enabled. Current SGLang: 95%, BF16 states, prefix caching disabled, prefill chunk 2,048; server cap 32 for 4B and 16 for 27B. Configuration differences remain in engine comparisons.

PR = #57962; PR2 = #52297 merged onto the same main base. New PR2 c=1,7,16,32 sweeps with prefix caching disabled were queued at migration time; no completed results were available. Existing PR2 values below are the earlier runs.

## 4B output tok/s

| Concurrency | Main | PR | PR vs main | PR2 | PR2 vs main | SGLang | active reqs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 523.6 | 761.8 | +45.5% | 717.4 | +37.0% | 879.9 | Same as concurrency |
| 7 | — | — | — | — | — | — | Not measured |
| 8 | 2,311.6 | 2,502.7 | +8.3% | 2,841.7 | +22.9% | 3,585.8 | SGLang: 9 |
| 16 | 2,116.0 | 2,284.9 | +8.0% | — | — | 4,551.7 | SGLang: 17 |
| 32 | 3,324.5 | 3,564.4 | +7.2% | — | — | 6,935.9 | Same as concurrency |

## 4B ITL p99 (ms)

| Concurrency | Main | PR | PR vs main | PR2 | PR2 vs main | SGLang | active reqs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 12.83 | 9.04 | -29.5% | 8.06 | -37.2% | 10.13 | Same as concurrency |
| 7 | — | — | — | — | — | — | Not measured |
| 8 | 61.66 | 61.99 | +0.5% | 47.66 | -22.7% | 30.37 | SGLang: 9 |
| 16 | 68.42 | 66.63 | -2.6% | — | — | 49.02 | SGLang: 17 |
| 32 | 80.91 | 74.42 | -8.0% | — | — | 41.45 | Same as concurrency |

## 27B output tok/s

| Concurrency | Main | PR | PR vs main | PR2 | PR2 vs main | SGLang | active reqs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 313.8 | 340.5 | +8.5% | 336.9 | +7.3% | 396.2 | Same as concurrency |
| 7 | — | — | — | — | — | — | Not measured |
| 8 | 1,171.8 | 1,263.1 | +7.8% | — | — | 1,719.7 | Same as concurrency |
| 16 | 1,465.7 | 1,449.0 | -1.1% | — | — | 2,662.4 | Main: 14; PR: 14 |
| 32 | 1,460.5 | 1,645.0 | +12.6% | — | — | 2,661.9 | Main: 14; PR: 14; SGLang: 16 |

## 27B ITL p99 (ms)

| Concurrency | Main | PR | PR vs main | PR2 | PR2 vs main | SGLang | active reqs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 25.01 | 20.62 | -17.6% | 20.31 | -18.8% | 21.71 | Same as concurrency |
| 7 | — | — | — | — | — | — | Not measured |
| 8 | 135.84 | 131.87 | -2.9% | — | — | 67.49 | Same as concurrency |
| 16 | 117.09 | 128.75 | +10.0% | — | — | 77.62 | Main: 14; PR: 14 |
| 32 | 116.21 | 109.47 | -5.8% | — | — | 77.41 | Main: 14; PR: 14; SGLang: 16 |

Active reqs show observed maxima only where they differ from client concurrency. SGLang's gauge occasionally exceeds client concurrency by one; values are preserved as reported.

Measured/warmup requests: c=1 100/10, c=7 140/14 (pending), c=8 160/16, c=16 320/32, c=32 640/64. Single runs; no uncertainty estimates.

Sources: [main/PR/PR2 artifacts](results/pr57962-sweep/), [current SGLang artifacts](results/sglang-no-prefix-95-bf16-prefill2048/), [revisions](revisions.json). Regenerate with `python scripts/render_results.py`.
