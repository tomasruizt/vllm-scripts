# Task: does adaptive verification beat a batch-size-keyed K schedule?

Read `README.md` in this folder first, then the 2026-10-19/20 sections of `../FINDINGS.md`.

## Question

Adaptive verification (AV, PR #60229) picks a verification budget every step from per-request drafter confidences and a cost table. vLLM already ships a much simpler policy, dynamic speculative decoding: a fixed K per batch-size range (`num_speculative_tokens_per_batch_size` in `vllm/config/speculative.py`, looked up by batch size in `vllm/v1/core/sched/scheduler.py`, see RFC #54749). On our workload a tuned fixed K is already close to AV with corrected costs (c128: fixed k=5 10,060 tok/s vs AV with measured costs 10,218 vs AV with its startup table 9,252, `rh-a8g`). If AV only matches the best K per batch size, the simpler policy may be the better product. This is an empirical question.

## Arms

Same frozen base `888ec6bb`, Qwen3.5-35B-A3B BF16 with the DFlash drafter, one B300, extended CUDA graph capture sizes (`cg_sizes` in `../run-steady.sh`) in every arm, steady harness, 3 repeats, c1, c8, c32, c64, c128 (c256 on TP2 optional, see `rh-fll`).

1. **Fixed K**: AV off, k ∈ {1, 2, 3, 5, 7}. Gives the best static K per concurrency. Note that DFlash's drafter work also scales with k.
2. **Dynamic K**: AV off, `num_speculative_tokens_per_batch_size` set from the best K per batch-size range. Tune the schedule on one workload and evaluate it on the other (below), or use two disjoint prompt folds, so it is not fitted to its own test data.
3. **AV, fixed profile**: AV on with the GDN profiling fix from fork PR tomasruizt/vllm#21 (null GDN state slots and mixed tails) applied to the frozen base. Check that the startup curve no longer shows the 512→1,024-token dip.
4. **AV, measured costs** (optional): the composition-aware measured cost model from `rh-dte`, if it is ready.

Existing cells that already match this setup (startup-table AV, fixed k=3/5/7, measured-cost AV) are in `../results/cost-curve-analysis-20261019/` and `../results/fixed-drafter-analysis-20261019/`. Reuse them only if harness, flags, capture list and GPU pairing match. Otherwise rerun in one session.

## Workloads

- **SPEED-Bench throughput-1K** (current workload, `../steady-speedbench.yaml`).
- **A heterogeneous-acceptance workload**, where per-request acceptance differs strongly (e.g. a mix of SPEED-Bench categories with very different acceptance, such as code and math against open-ended chat). AV's per-request adaptivity should help most here, so this is the fair test of its value. Report per-category AL to show the heterogeneity.

## Checks first

- Confirm dynamic SD works with the DFlash drafter and AV off on this base (graph capture with the per-batch-size K, correct verification lengths). If it does not, record why and fall back to per-concurrency fixed-K servers as an upper bound for a static schedule.
- Use distinct ports per concurrent run (`PORT_A`/`PORT_B`) and never start servers from inside a checkout.

## Deliverable

- Table per workload and c: throughput / TPOT / AL with min–max over repeats for every arm, plus verified drafts per step for the AV arms.
- The tuned dynamic-K schedule and how it was chosen.
- Verdict:
  - AV beats dynamic K on the heterogeneous workload but not on SPEED-Bench: per-request adaptivity has value where acceptance varies.
  - AV is within noise of dynamic K on both: recommend dynamic K (simpler, already in main) and say what AV would need to win.
  - Dynamic K beats AV: identify whether pricing (cost table) or estimation (confidences) is the limit, using the verified-draft traces.
