# AV budget objective: per-step ratio vs one shared bar

Question: AV picks its draft budget per step with `argmax_b (n + G(b)) / C(b)`. Long-run throughput is maximized by one shared bar, `argmax_b (n + G(b)) - lam * C(b)` with `lam` the long-run tokens per ms (Dinkelbach). Is the shared bar worth implementing?

**Answer (2026-11-25): no.** At steady load the ideal (non-causal) shared bar gains a median of +0.0% and at most +1.2% modeled throughput over 27 workloads, and a causal EMA `lam` gains the same or less. The hypothesis is right in direction (prefill steps over-verify, decode-only steps under-verify), but the effect is second order. Phase 3 (implementation, online A/B) was not run, per the stop rule.

Method: offline replay on the verify-all traces of [av_alloc_headroom](../av_alloc_headroom) (Qwen3-4B and Qwen3.8-27B with DFlash2, B200, temp 0, vLLM `89e860de00` from 2026-11-17 plus the trace patch). Run `make analyze`, it needs only the data release.

## Phase 0: the offline budget reproduces production

Offline `get_num_tokens` equals the logged `policy_budget` in 99.88% of 54,383 steps (13 traces, worst trace 98.87%, 12 of 13 at 99.2% or more). The trace patch does not log two inputs, so they are reconstructed, and the fidelity validates both:

- The stale host confidences the budget reads: each request's confidences of its previous traced step, 1.0 on its first step. Using the logged (fresh) confidences instead matches only 75 to 87% of steps.
- `n` (sampling requests): set to `num_reqs`.

## Phase 1: `r_t` varies, in the predicted direction

`r_t` is the step's own best ratio, `lam` the fixed point. Steady window (request count at 90% or more of its peak), main cost tables, mixed prompts, graphs covering all decode sizes, logits cap not binding:

| target | c | prefill steps | CV of r_t | mean r_t/lam, prefill | mean r_t/lam, decode-only | p5 to p95 of r_t/lam |
|---|---|---|---|---|---|---|
| 27B | 64 | 36% | 0.13 | 0.95 | 1.07 | 0.76 to 1.17 |
| 27B | 128 | 36% | 0.14 | 0.96 | 1.09 | 0.77 to 1.19 |
| 27B | 256 | 36% | 0.13 | 0.95 | 1.07 | 0.76 to 1.17 |
| 27B TP2 | 128 | 55% | 0.20 | 0.98 | 1.21 | 0.69 to 1.29 |
| 27B TP2 | 256 | 70% | 0.23 | 1.01 | 1.31 | 0.54 to 1.35 |
| 4B | 64 | 24% | 0.16 | 0.80 | 1.17 | 0.67 to 1.21 |
| 4B | 128 | 42% | 0.07 | 0.98 | 1.02 | 0.92 to 1.08 |
| 4B | 256 | 64% | 0.08 | 1.01 | 1.00 | 0.87 to 1.09 |
| 4B | 512 | 82% | 0.10 | 1.01 | 1.04 | 0.89 to 1.09 |

The bars differ by more than a few percent, so phase 2 ran.

## Phase 2: the gain is below 1%

Throughput gain over the per-step ratio in %, realized tokens over modeled cost, same steps, steady window. Brackets are block-bootstrap 95% intervals over steps. `bar` uses the offline fixed-point `lam`, `ema100` a causal `lam` (EMA of realized tokens over cost, 100 steps).

| target | c | main: bar | main: ema100 | linear: bar | wall: bar |
|---|---|---|---|---|---|
| 27B | 64 | +0.06 [+0.02, +0.11] | +0.05 | +0.05 | +0.61 [+0.47, +0.76] |
| 27B | 128 | +0.02 [+0.01, +0.04] | +0.02 | +0.07 | +0.71 [+0.57, +0.83] |
| 27B | 256 | +0.09 [-0.02, +0.31] | +0.09 | +0.05 | +0.52 [+0.44, +0.59] |
| 27B TP2 | 128 | +0.36 [-0.02, +1.13] | +0.35 | +0.43 | n/a |
| 27B TP2 | 256 | +1.01 [+0.40, +1.85] | +1.00 | +1.23 | n/a |
| 4B | 64 | +1.23 [+0.83, +1.41] | +1.28 | +1.26 | -0.05 [-0.10, -0.01] |
| 4B | 128 | -0.18 [-0.31, -0.06] | -0.13 | -0.23 | +0.04 [-0.00, +0.09] |
| 4B | 256 | -0.03 [-0.04, -0.02] | -0.01 | +0.08 | +0.05 [-0.02, +0.10] |
| 4B | 512 | -0.00 [-0.01, +0.01] | +0.00 | +0.63 | +0.09 [+0.06, +0.11] |

Cost models:

- `main`: the logged tables (staircase below the graph limit).
- `linear`: the same table, interpolated between captured sizes.
- `wall`: `const + a * num_reqs + b * prefill_tokens + slot_price * budget`, fitted on the gaps between trace timestamps of the policy and the verify-all server (R² 0.94 to 0.99). This has the form of [vllm#62960](https://github.com/vllm-project/vllm/pull/62960), a base plus a linear slot price. The real #62960 tables need their own profiling pass and were not obtained.

Over all 27 workloads (including single-dataset and default-graph-limit runs), steady window: `bar` median +0.02% (main), +0.08% (linear), +0.05% (wall), max +1.2%, +2.5%, +0.8%. The +2.5% is 4B c128 with the default 1,024-token graph limit, where steps run eager.

Breakdown, 27B TP2 c256 (main), share of scheduled drafts verified:

| step type | share of cost | per-step ratio | shared bar |
|---|---|---|---|
| decode-only | 23% | 53% | 58% |
| with prefill, within graphs | 63% | 57% | 58% |
| with prefill, above the graph limit | 14% | 84% | 60% |

The shared bar moves budget from prefill steps to decode-only steps as predicted, but the moved slots sit close to the bar, so each is worth little. The largest shift is in steps that a prefill pushes past the graph limit.

Why the toy example does not carry over: its two steps differ 4x in `r_t`. Real steady-state steps differ by 10 to 30%, and the loss is quadratic in that error.

## Over the whole workload the replay overstates the gain

Including ramp-up and ramp-down, `bar` shows a median +1.2% and up to +2.8% (main). This is an artifact. In the ramp-down few requests remain, `r_t` is far below the global `lam`, and the shared bar cuts their verification. The replay credits the saved time at the global rate, but the remaining requests need those tokens anyway and produce them at the local rate, so the tail gets longer. Load is exogenous: the right `lam` is the one of the current load level, and the per-step ratio already tracks that. A slow EMA (100 steps or more) would under-verify after every load drop. This is the stability risk of the idea.

## The cost model matters more than the objective

- The 4B tables overstate the step time about 3x (modeled 65 ms vs 18 ms between timestamps, decode-only steps at c256/c512). The single-GPU 27B tables are within 8 to 25%.
- Under the wall-clock fit, the ratio rule would verify 50 to 59% of drafts where the main tables verify 68 to 96% (27B c64 to c256, 4B c128 to c512), worth +4 to +15% wall-clock-modeled throughput (4B c64 is an outlier: the staircase verifies only 27%). That is roughly 10x the effect of the objective. The estimate is circular (fitted and scored on the same linear model), so it is a pointer and not a result. It agrees in sign with the E2E finding that AV over-verifies at high concurrency ([av_chunk_cap_e2e](../av_chunk_cap_e2e)).

## Assumed, not verified

- The replay assumes the same drafts regardless of the budget. Nothing was confirmed online.
- Stale confidences and `n` are reconstructed (see phase 0), 0.12% of budgets differ.
- The wall-clock model is a 4-parameter linear fit of timestamp gaps, with no graph-limit jump. The TP2 run has no policy server, so it has no wall-clock model.
- The #62960 cost form is represented by proxies (`linear`, `wall`), not by its tables.
- All traces are at temperature 0, closed-loop AIPerf with 4 requests per concurrency slot.
- Workloads where the 1 GB logits cap binds (4B c256 and c512 of the `cg` run) are excluded.

## Data release fix

`package_data.py` of av_alloc_headroom wrote the first of the two cost-table records a server emits (from the memory-profiling capture). The budget uses the second. This is fixed in the script and in the local `results/data-release`, the published zip still holds the old tables.
