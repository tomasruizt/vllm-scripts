# What does the stale budget input cost adaptive verification?

Task brief: `av-stale-confidence-investigation.md` (2026-11-25). Offline replay only (Phase 1), no new GPU runs.

## Answer

Staleness costs almost nothing where AV is normally used, and no cheap mitigation recovers the rest. With 33 or more running requests the stale budget loses 0.0 to 0.2% modeled throughput against a budget sized on fresh confidences. The loss grows only in small batches: 0.4% at 17 to 32 running requests and 1.3% at 16 or fewer (0.8 to 2.9% across the 7 runs), under the cost model of vLLM main. Under a linear slot price (proxy for vllm#62960) it is 0.4% at 16 or fewer requests and under 0.3% elsewhere. I stopped after Phase 1: the loss is real per-step information that only the fresh values hold, so the host-side mitigations (better fill, smoothing) recover nothing, and the split decision cannot help under the staircase cost.

## The lag (confirmed)

- The budget of step t is computed from the confidences of the drafts verified at step t-1. The kernel ranks with the confidences of the drafts verified at step t. Lag: exactly 1 step with the default 2-slot ring.
- A request's first verification step reads 1.0 at every position.
- Evidence: reimplementing `get_num_tokens` offline with "each request's `p` from the previous traced step, 1.0 if absent" reproduces the logged `policy_budget` in 99.91% of 23,439 steps (normal AV runs) and 99.84% of 21,752 steps (verify-all runs). The fresh values reproduce it in only 79 to 80% of steps, lag 2 in 79 to 80%, lag 3 in 77 to 79%.
- The trace field `p` is the fresh GPU tensor (`_confidence_probs[idx_mapping]`, cloned in `reallocate_drafts`).
- The 0.1 to 0.2% unmatched steps are prefill-heavy (3,400 to 5,200 prefill tokens against ~100 on average), consistent with the one input the trace lacks (`num_sampling_requests`, assumed equal to `num_reqs`).
- On main (`99d66fffac`, 2026-11-25) the ring has `max(2, max_concurrent_batches)` slots, and `max_concurrent_batches = pp_size + async_scheduling_depth - 1` (default depth 2, from #64145). `record_confidences` rotates one slot per step, so the lag is `slots - 1` steps. Lag 1 at the defaults. Larger depths were read from the code, not traced.

## How different are stale and fresh inputs?

- Per request, the expected accepted drafts of consecutive steps correlate at r = 0.33 to 0.64. One step back is already a weak predictor, and lag 2 or 3 is barely worse than lag 1.
- Per batch, the sum of the survival curve moves by 24% (mean absolute change) at 16 or fewer requests, 9% at 17 to 32, 6% at 33 to 64 and 3% above 128. Averaging over requests is what makes the stale curve a good stand-in in large batches.
- The stale input is biased up by 0.4 to 1.2% (the 1.0 fill plus drift).

## How often does B change, and what does it cost?

Stale vs. fresh budget, verify-all traces, valid workloads. `differ`: share of steps with a different B. `abs ΔB`: mean absolute change as a share of the scheduled drafts. Throughput is realized tokens per modeled cost.

| running requests | steps | differ (table) | abs ΔB (table) | tput, table cost | tput, linear cost | hindsight B, table cost |
|---|---|---|---|---|---|---|
| 1 to 16 | 6,937 | 37% | 5.5% | **-1.27%** | -0.42% | +3.3% |
| 17 to 32 | 3,027 | 17% | 5.7% | -0.42% | -0.26% | +0.5% |
| 33 to 64 | 7,710 | 10% | 1.7% | -0.07% | -0.14% | +0.1% |
| 65 to 128 | 1,776 | 13% | 3.1% | -0.20% | -0.09% | +0.3% |
| 129+ | 2,302 | 9% | 0.8% | -0.02% | -0.03% | +0.1% |

- Under the staircase cost B snaps to graph sizes, so it changes rarely but by a whole bucket (p99 change: 22 to 77% of the drafts). Under the linear price it changes in 89 to 99% of steps, by 1 to 5% of the drafts.
- The change is unbiased (signed ΔB within 0.6% of the drafts): stale adds noise to B, it does not over or under-verify.
- The normal AV traces give the same change rates (36%, 17%, 9%, 18%, 10%), so verify-all did not distort this.
- By workload (table cost): the only cells at 1% or more are Qwen3-4B at c16 (HumanEval -1.17%, math_reasoning -1.09%, writing -1.35%, ~15 running) and Qwen3-4B mixed c32 (-1.03%, 21 running). Qwen3.8-27B is at most -0.24% in every cell. No dataset effect beyond batch size.
- Steps cut by the 1 GB logits cap (4B mixed at c256 and c512 without the lifted cap) show -0.01 to -0.02%: staleness is irrelevant there, as expected.

## Where does the error come from?

Throughput vs. the fresh budget, table cost.

| budget input | 1 to 16 | 17 to 32 | 33 to 64 | 65 to 128 | 129+ |
|---|---|---|---|---|---|
| stale (production) | -1.27% | -0.42% | -0.07% | -0.20% | -0.02% |
| drift only (stale, new requests get fresh values) | -1.28% | -0.41% | -0.07% | -0.18% | -0.02% |
| fill only (fresh, new requests get 1.0) | +0.02% | -0.02% | 0.00% | 0.00% | 0.00% |
| stale, 1.0 fill replaced by the batch mean | -1.34% | -0.41% | -0.06% | -0.13% | -0.02% |
| lag 2 | -1.36% | -0.48% | -0.10% | -0.21% | -0.02% |
| lag 3 | -1.47% | -0.46% | -0.11% | -0.27% | -0.01% |
| moving average of the stale curve, α = 0.3 | -1.44% | -0.42% | -0.08% | -0.33% | -0.01% |
| moving average of the stale curve, α = 0.05 | -1.87% | -0.75% | -0.16% | -0.48% | -0.08% |

- All of the loss is step-to-step drift. The 1.0 fill costs nothing here, because only 0.6% of the rows are filled (max 1,024 output tokens).
- Row mismatches do not exist in practice: 15 of 1.5M rows were filled although the request had been seen before.
- Smoothing makes it worse, so the stale snapshot does carry some information. A longer lag costs little extra, so a deeper async pipeline is not a concern.

## When is it worst?

- Small batches (32 or fewer running requests), where one batch is too few requests to average out the drift, and where the budget matters most (hindsight B: +3.3%).
- The staircase cost model, where a small input change moves B by a whole graph bucket.
- Possibly short outputs (not in these traces). Filling a random 10% of the rows of the fresh input with 1.0 costs 0.4 to 1.2% at 17 to 128 running requests (25%: 1.3 to 3.1%) under the table cost, and under 0.5% under the linear cost. The batch-mean fill removes it (within ±0.25%). 10% filled rows corresponds to requests that live ~10 steps, i.e. ~30 to 40 output tokens.

## Recommendation

- No change for the stale budget. The only setting with a loss of 1% or more is 16 or fewer running requests under main's cost model, and it needs the fresh values, which the host cannot read without a device sync.
- Mitigation 3 (host picks the graph bucket, GPU admits by price) cannot help under the staircase cost, because the cost is the bucket. Under a linear price the whole loss is 0.4% or less.
- The batch-mean fill (mitigation 1) is a one-line change that only pays for short-output workloads. It needs a real short-output trace before an A/B. Closed-loop clients with a fixed `max_tokens` would synchronize the arrivals and overstate it.

## Assumed, not verified

- Costs are AV's own startup-profiled tables, not measured step times. The replay assumes the drafts do not depend on the budget.
- `num_sampling_requests` is taken as `num_reqs` (not in the trace).
- The linear cost keeps the logged base cost and prices a slot at the mean slope of the logged verify table (`linear`) or a quarter of it (`linear_cheap`, tput loss 0.11% or less in every bin). These are not the prices vllm#62960 profiles.
- The lag for `async_scheduling_depth > 2` or pipeline parallelism.
- The short-output result is synthetic (random rows, fresh input).
- Scope of the traces: temperature 0, B200, Qwen3-4B and Qwen3.8-27B with DFlash2 drafters and the online logistic estimator, vLLM `89e860de00` (main on 2026-11-17) plus the trace patch.

## Reproduce

`make analyze` runs [stale.py](stale.py) on the data release of speculators#1430 (45 s) and prints the tables of [summary.sql](summary.sql). Outputs: `results/steps.parquet` (one row per step, cost model and budget input) and `results/requests.parquet`.

The released `*_cost_tables.json` held the first of the two cost records the server emits at startup. The second one is in effect (with the first, the replay matches only 18 to 61% of the steps). `av_alloc_headroom/package_data.py` now exports the second. The headroom results did not use the tables.

Serve command of the runs (model, drafter, K, TP and `max_num_seqs` vary, the exact command is next to each trace):

```
vllm serve Qwen/Qwen3-4B --max-model-len 16384 --max-num-seqs 512 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.85 --no-enable-prefix-caching --speculative-config '{"model":"RedHatAI/Qwen3-4B-speculator.dflash2","num_speculative_tokens":8,"method":"dflash","enable_adaptive_verification":true}' --compilation-config '{"max_cudagraph_capture_size": 4608}'
```
