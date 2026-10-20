# Task: does a corrected cost table make AV choose better draft counts?

Read `README.md` in this folder first.

## Context

AV (PR #60229) picks a per-step draft budget `b` by maximizing expected accepted tokens per millisecond:

`b = argmax (n_sampling + Σ top-b survival scores) / (draft_cost[n_reqs] + verify_cost[n_non_draft + b])`

(`get_num_tokens` in `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py`). Both tables come from a startup profile of synthetic dummy batches (`batches_to_profile`, `set_initial_cost_curves`, `set_cost_curves`, `build_cost_tables_from_curves`). For decode steps the verify index is `n_reqs + b` (one bonus token per request plus admitted drafts).

We measured that this table is wrong in the direction that over-prunes. It overprices extra drafts by ~1.3× at c128 and ~2.5× at c32, misses ~4 to 5 ms of per-step overhead, and has shape artifacts (a dip past 512 tokens hidden by a non-decreasing clamp). It also prunes at low concurrency, where verification is nearly free: AL at c1 is 2.61 with AV vs 2.71 without. See section 6 of `../av-cost-model.html` and `../plots/cost-curve.png`.

Measured decode steps behave roughly as `step ≈ fixed + per-request × R + per-token × T` (R requests, T = R + verified drafts). A fit over ~10K AV-off steps gave ~8.5 ms + 50 µs × R + 13 µs × T, but the curve is concave, so fit per request count instead of one global line. This model maps onto AV's two tables: `draft_cost[R] = fixed + drafter + per_request(R)` and `verify_cost[T] = per_token × T`.

Before this matters, prefill steps must be graphed (otherwise eager prefill steps dominate the run and dilute any decode-side gain). All arms below use the extended capture sizes (the `*cg` modes in `../run-steady.sh`, `cg_sizes`).

## Arms

All with extended capture sizes, the steady harness (`../run-steady.sh`, `../steady-speedbench.yaml`), cells c1, c8, c32, c64, c128, 3 repeats, AV off and on alternating across two GPUs as in the existing runs. Use free ports (`PORT_A`/`PORT_B`).

| arm | config | status |
|---|---|---|
| A | AV off, k=7 | done: `~/runs/qwen-speedbench-steady-cg-20261019` (`offcg`) |
| B | AV on, startup table | done: same run (`oncg`) |
| C | AV on, corrected table | to run |
| D | AV off, fixed k=3 and k=5 | to run |

Arms A and B can be reused only if C and D use the identical harness, server flags and capture list. Otherwise rerun A and B in the same session.

## Steps

1. **Calibrate.** Run traced AV-off servers with k=1, 3, 5 (k=2 and k=7 traces exist in `~/runs/qwen-av-k2-control-20261018` and `~/runs/qwen-av-steps-20261019`, but redo them with the extended capture sizes so all calibration comes from one setup) at c8 to c128 using the step trace (`../av-diagnostics.patch` + `../av_trace.py`, `*diag` modes, `AV_TRACE_PATH`). Each `step` record has the executed shape (`num_reqs`, `num_tokens`, `num_draft_tokens`, `prefill_tokens`). Step time is the gap to the next step record, so use medians per shape, decode-only steps only (`prefill_tokens == 0`). Fit `fixed`, `per_request(R)` and `per_token`. The drafter's cost belongs in `draft_cost` (check whether the step gap already includes it).
2. **Inject.** Write a small patch (on a copy of the checkout, ~20 lines) that loads `draft_curve` and `verify_curve` from a JSON file named by an env var (e.g. `VLLM_AV_COST_CURVES`) instead of the startup profile. Note `build_cost_tables_from_curves` treats sizes up to the capture limit as a step function over the profiled points and clamps to non-decreasing. Either supply a point at every size AV can index, or inject the final tables directly. Log the injected tables with the existing `costs` trace record to verify.
3. **Run C and D** (3 repeats). Keep the step trace on in C so verified drafts per step are recorded. If tracing costs measurable throughput, run untraced repeats for the throughput numbers and one traced repeat for the decisions.
4. **Analyze** with `../build_report.py` (`--run` per arm, `--mode-suffix`) and duckdb on the step traces.

## Deliverable

- Table per c: throughput, TPOT and AL for A, B, C, D(k=3), D(k=5), with spread across repeats.
- Verified drafts per decode step for B and C per c, and whether C verifies all drafts at c1 to c8.
- The fitted coefficients and the injected tables, plus the patch.
- Verdict:
  - C beats B at c≥32 and matches the best fixed k at every c: cost-curve quality matters, propose an upstream fix (relates to #54065).
  - C ≈ B within noise: the table's errors do not change outcomes on this model.
  - Fixed k beats both: the limit is AV's objective or its acceptance estimate, not the table.
