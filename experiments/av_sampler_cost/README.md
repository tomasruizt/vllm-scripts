# Does the untimed `sample()` phase matter for the AV cost model?

**Answer:** On dense models, barely. Pricing `sample()` on top of vllm#62960 moves throughput by -1% to +1% on Qwen3-4B and Qwen3.8-27B. On the Qwen3.6-35B-A3B MoE it gains 2 to 6%, but mostly because #62960 underprices the MoE forward by 10 to 20x, and any extra price per slot moves the budget in the right direction. The larger error is the forward price of #62960 on the 27B and the MoE, not the missing `sample()` term.

Naming: "#62960" is vllm#62960 merged with main 57906d3cc0 (2026-11-25). "Our branch" is #62960 plus a `sample()` price that is profiled at startup. All numbers are from one B200, with `temperature=0.7, top_p=0.8, top_k=20` unless a row says otherwise.

## Price of one more verified slot

Measured on real serving steps with a uniform random AV budget per step, decode-only steps, GPU event timing.

| model | #62960 forward price | real forward | real LM head | real sampler |
|---|---|---|---|---|
| Qwen3-4B (dflash, K=8) | 5.7µs | 5.0 to 5.9µs | 0.5µs | 1.8µs |
| Qwen3.8-27B (dflash, K=7) | 21.3µs | 41 to 56µs | 1.1µs | 3.3µs |
| Qwen3.6-35B-A3B (dspark, K=7) | 1.0µs | about 21µs | 0.7µs | 3.2µs |

Sampler price per row by sampling mode (Qwen3-4B, c128): greedy 0.1µs, temperature only 0.8µs, top-p only 2.0µs, top-k+top-p 1.8µs, with presence penalty 2.2µs. One sampling request in a greedy batch switches the whole batch to the logits-processing path (+0.5µs per row on the 4B).

Kernel profile of `sample()` (Qwen3-4B, c128, about 900 rows, top-k+top-p, 2.3ms in total): top-k/top-p kernel 44%, LM head GEMM 23%, FP32 copy of the logits 18%, temperature 7%, rejection sampling under 2%.

## Worst case for #62960 (Qwen3-4B, c256, about 1,690 logits rows)

| mode | #62960 step cost | our branch | real step |
|---|---|---|---|
| greedy | 20.6ms | 21.4ms | 22.7ms |
| top-k+top-p | 20.6ms | 24.0ms | 24.8ms |
| top-k+top-p, presence penalty | 20.7ms | 24.1ms | 25.9ms |

#62960 misses 17 to 20% of the step under sampling. Our branch misses 3 to 7%.

## Throughput: our branch against #62960

4 interleaved pairs of 30s on one server (A/B/B/A), AIPerf, 512 output tokens, mixed prompts.

| model | c | #62960 tok/s | our branch tok/s | gain (min to max) | drafts verified |
|---|---|---|---|---|---|
| Qwen3-4B | 32 | 18,137 | 18,104 | -0.2% (-1.0 to +0.9) | 63% to 60% |
| Qwen3-4B | 128 | 29,987 | 29,927 | -0.2% (-0.9 to +0.2) | 51% to 47% |
| Qwen3-4B | 256 | 33,971 | 34,226 | +0.8% (+0.5 to +1.2) | 60% to 56% |
| Qwen3.8-27B | 16 | 2,678 | 2,672 | -0.2% (-2.0 to +0.9) | 74% to 71% |
| Qwen3.8-27B | 64 | 4,406 | 4,435 | +0.7% (+0.2 to +1.0) | 57% to 54% |
| Qwen3.6-35B-A3B | 32 | 6,635 | 6,769 | +2.0% (+1.4 to +2.9) | 83% to 69% |
| Qwen3.6-35B-A3B | 128 | 12,949 | 13,717 | +5.9% (+5.6 to +6.2) | 76% to 59% |
| Qwen3.6-35B-A3B | 256 | 13,222 | 13,933 | +5.4% (+5.1 to +5.8) | 76% to 59% |

For reference, #62960 against main (single 20s cells, top-k+top-p): Qwen3-4B +30% at c32, +8% at c128, +7% at c256. Qwen3.8-27B +4% at c16, +7% at c64. Main prices a slot at about 160µs on the 4B when the batch is below `max-num-seqs`, against 5µs real.

## The fix on our branch

`get_num_tokens()` adds `fixed_ms + row_ms * (num_reqs * num_bonus_tokens + budget)` to the cost of every candidate budget. There are two (fixed, per row) pairs: one for batches where no request needs logits processing, one for the rest. Both are measured once at startup, right after the forward profile of #62960, by timing `sample()` on a small and a large dummy verification batch.

Three details turned out to matter for the startup profile:

* The dummy run's hidden states are not usable. They give NaN logits on the 27B and the MoE (both have GDN layers), and top-k/top-p is about 3x slower on NaN rows. They also make the LM head about 25% faster than in serving on the 4B. The profile feeds `sample()` random hidden states instead, which lands within 10% of the real price on the 4B (2.38µs against 2.2µs per row).
* `sample()` is timed queued behind a dummy forward. On an idle GPU the small batch is bound by host time, which lowers the price per row.
* Forcing an acceptance length on the dummy requests is unnecessary. Accepting 2 or all 8 drafts moves the price by under 0.1µs per row.

An online least-squares fit of the price from live steps also works (2.28µs per row on the 4B) but is harder to defend, and was dropped.

## Recommendation

1. Do not carry the `sample()` term for its own sake. It is correct, but worth about 1% at most on dense models.
2. Report the forward price error to #62960: 2x low on Qwen3.8-27B and 10 to 20x low on Qwen3.6-35B-A3B. Random input tokens in the forward profile (so that MoE tokens do not all route to the same experts) only move the slot price from 1.0 to 1.9µs on the MoE and from 21.3 to 27.8µs on the 27B, so that is not the main cause.
3. Once the forward price is right, re-measure the `sample()` term on the MoE. With random inputs in the forward profile the gain of our branch already drops to +1.3% (c32), +2.5% (c128) and +3.3% (c256).

## Unverified

* Why #62960's slot price is low on the 27B and the MoE. #62960 on its own base cannot run the 27B with AV (GDN backend unsupported there), so a merge artifact is not excluded. Its KV price table is zero on the merge.
* The real MoE forward price (21µs per slot) comes from one run. It agrees with the forward time difference between the two A/B arms (about 19µs per slot).
* Random hidden states give flat logits (top probability near 0). Real logits are peaked. The price matched on the 4B, the 27B (4.8µs against about 4.4µs) was only measured up to 512 rows in serving.
* Only one GPU type (B200), one workload, thinking enabled, no structured outputs in the A/B.
* #66792 (Triton top-k/top-p kernel) would roughly halve the sampler price and was not measured here.

## Reproduce

```bash
vllm serve Qwen/Qwen3.6-35B-A3B --revision=995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --max-model-len 16384 --max-num-seqs 256 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.85 --no-enable-prefix-caching \
  --speculative-config '{"model":"RedHatAI/Qwen3.6-35B-A3B-speculator.dspark","num_speculative_tokens":7,"method":"dspark","enable_adaptive_verification":true}' \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --compilation-config '{"max_cudagraph_capture_size": 2048}'

aiperf profile --model Qwen/Qwen3.6-35B-A3B --endpoint-type chat --streaming \
  --input-file prompts/mixed.jsonl --custom-dataset-type single_turn --dataset-sampling-strategy sequential \
  --concurrency 128 --benchmark-duration 30 --benchmark-grace-period 60 \
  --extra-inputs '{"max_tokens":512,"temperature":0.7,"top_p":0.8,"top_k":20}'
```

The scripts need a vLLM tree with the step trace patch (`AV_STEP_TRACE_PATH`, `AV_STEP_TRACE_CONTROL`). It lives in two local vLLM branches that are not pushed: `av-sampler-cost-trace` (main plus trace) and `av-sampler-cost-62960-on-main` (our branch).

`make launch RUN_ID=... GPU=... EXTRA_ENV='BUDGETS=pr62960,ours REPEATS=4 CELL_S=30 ...'` runs the A/B on one server (the budget mode is switched per cell through a control file), `make fetch-results` and `make analyze` build `results/*.parquet`. `make launch-profile` plus `prof_sample.py` give the kernel attribution.
