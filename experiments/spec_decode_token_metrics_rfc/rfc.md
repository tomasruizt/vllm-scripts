# [RFC]: Consistent semantics for drafted, verified and accepted spec decode tokens

## Summary

vLLM reports how many draft tokens were drafted and accepted during speculative decoding (SD). With adaptive verification (AV) and structured outputs (SO), these numbers no longer mean the same thing on every code path. We propose three clear definitions, one invariant that always holds, and to report all three metrics when using SD.

| Count | Meaning |
| --- | --- |
| **Drafted** | Draft tokens proposed by the drafter. |
| **Verified** | Draft tokens the target model ran a forward pass on. |
| **Accepted** | Draft tokens the target model accepted. |

Invariant: `drafted >= verified >= accepted`, in total and at every step.

This RFC grew out of reviewing #58340, and discussions with @benchislett and @yzong-rh.

## Background

Two features make "drafted" and "verified" differ.

**Adaptive verification.** The drafter proposes `k` tokens, but the target only verifies a prefix of them, chosen by a confidence score. This saves target compute when the later draft tokens are unlikely to be accepted. The skipped draft tokens were proposed, but never verified.

**Structured outputs.** A grammar (e.g. a JSON schema) constrains the output. A draft token that violates the grammar can never be accepted. When vLLM checks the grammar depends on the scheduling mode:

- **Sync scheduling** checks the draft tokens against the grammar before the forward pass. Invalid draft tokens are dropped and never included in the target forward pass.
- **Async scheduling** launches the forward pass first and checks the grammar on the CPU in parallel, to keep the GPU busy. Invalid draft tokens are included in the target forward pass, then masked out before sampling, so they are always rejected.

## Problem

Today the counts depend on the code path, not only on what happened:

- **Async scheduling** subtracts grammar-invalid draft tokens from `num_draft_tokens`, even though they were included in the target forward pass.
- **Sync scheduling** never counts grammar-invalid draft tokens at all, because they are dropped before scheduling.
- **Verified tokens are not reported.** With AV, there is no way to see how many draft tokens the target actually verified.

So two runs with identical draft tokens can report different numbers, which is confusing for anyone reading the metrics.

## Proposal

### 1. Drafted means proposed by the drafter

`num_draft_tokens` counts every token the drafter proposed, including:

- draft tokens that violate the grammar (both scheduling modes),
- draft tokens skipped by AV.

Padding slots are a special case. They count if the target runs a forward pass on them, and are excluded otherwise. For example, when `ngram_gpu` finds fewer than `k` matches, the V2 model runner fills the empty slots with real tokens and verifies them, so they count. The V1 model runner pads with `-1` and trims those slots before the forward pass, so they don't. This preserves the invariant `drafted >= verified`.

### 2. Verified means included in the target forward pass

`num_verified_tokens` counts draft tokens the target model ran a forward pass on. It measures how much target compute went into verification.

- With AV, the model runner reports it, because only the model runner knows which draft tokens it verified.
- Otherwise, the scheduler computes it as `min(scheduled draft tokens, proposed draft tokens)`. This counts grammar-invalid draft tokens in async scheduling (included in the target forward pass), and drops them in sync scheduling (not included).

### 3. Report all three everywhere

Verified is reported wherever drafted and accepted are: Prometheus, the periodic log line, the per-request metrics, `vllm bench serve` and the Rust frontend. See [appendix A](#appendix) for the field names.

## Worked examples

All examples use `k = 4`.

| Case | What happens | drafted | verified | accepted |
| --- | --- | --- | --- | --- |
| No grammar, no AV | All 4 draft tokens in the target forward pass, target accepts 2 | 4 | 4 | 2 |
| AV | Drafter proposes 4, target verifies the first 3, accepts 2 | 4 | 3 | 2 |
| Sync scheduling, grammar rejects last 2 draft tokens | Only 2 draft tokens in the target forward pass, target accepts 1 | 4 | 2 | 1 |
| Async scheduling, grammar rejects last 2 draft tokens | All 4 in the target forward pass, last 2 masked, target accepts 1 | 4 | 4 | 1 |
| V2 `ngram_gpu`, only 2 matches | Empty slots filled with the last sampled token and verified normally | 4 | 4 | ≤ 4 |
| V1 `ngram_gpu`, only 2 matches, SO | `-1` padding is trimmed by the worker before the forward pass | 2 | 2 | ≤ 2 |

The acceptance rate (accepted / drafted) is the same for sync and async scheduling (1/4). Only verified differs, which reflects the real difference in compute.

See [appendix B](#appendix) for the edge cases behind these rows, e.g. how async draft tokens enter the forward pass before the grammar check, and how `-1` placeholders are handled.

## Impact on users and clients

Without SO, drafted and accepted do not change, and verified equals drafted unless AV is on.

With SO:

- `num_draft_tokens` goes up, and `draft_acceptance_rate` goes down. The new numbers are the same in sync and async scheduling.
- Clients that compute `accepted / draft_tokens` from Prometheus will see lower rates. This includes AIPerf's speed-bench report, the Datadog integration and many user scripts.
- AIPerf's per-request adapter keeps working. It ignores new fields, and all of its consistency checks still hold. Its docs say vLLM subtracts grammar-invalid draft tokens, which would need an update.
- The new Prometheus counter name contains neither `num_draft_tokens` nor `num_drafts`, so parsers that match metric names by substring (like `vllm bench serve`) do not double count.

## Out of scope

- **V1 `ngram_gpu` without SO.** The worker trims `-1` padding, but the scheduler never sees it. So drafted and verified still count all `k` slots. This is pre-existing. A fix would let the V1 worker report its valid draft token counts.
- **Drafts after an EOS.** Stats are recorded before the output is trimmed at the stop token. So accepted can include tokens after an accepted EOS, which are then discarded. This is tracked in #56101. Two fixes are discussed there: clip accepted at the stop token, or keep accepted and report committed tokens separately (#56137, #56195). Both keep the invariant, since they only change accepted or add a new count.

## Open questions

1. Is `accepted / drafted` the right default for `draft_acceptance_rate`, or should we also expose `accepted / verified` as its own field? See [appendix C](#appendix) for arguments.
2. Should draft tokens skipped by AV lower the acceptance rate? They were not rejected by the target, but a better drafter could have produced higher-confidence draft tokens.

## Appendix

<details>
<summary>A. Where each count is reported</summary>

| Surface | drafted | verified | accepted |
| --- | --- | --- | --- |
| Prometheus | `vllm:spec_decode_num_draft_tokens` | `vllm:spec_decode_num_verified_tokens` (new) | `vllm:spec_decode_num_accepted_tokens` |
| Periodic log line | `Drafted:` | `Verified:` (new) | `Accepted:` |
| Per-request metrics | `num_draft_tokens`, `per_step_drafted` | `num_verified_tokens`, `per_step_verified` (new) | `num_accepted_draft_tokens`, `per_step_accepted` |
| `vllm bench serve` | yes | new | yes |
| Rust frontend | yes | new | yes |

</details>

<details>
<summary>B. Edge cases in how draft tokens reach the target forward pass</summary>

- **Async draft tokens enter the target forward pass before the grammar check.** The forward pass for step N starts while the CPU still validates the draft tokens against the grammar. So invalid draft tokens cost target compute, even though they can never be accepted. This is why they count as verified on that path.
- **`-1` placeholders never reach the embedding layer as `-1`.** The V1 model runner clamps input ids to `>= 0` before the embedding lookup. This consumes target compute capacity. The rejection sampler sees the original `-1` and rejects that position outright.
- **`ngram_gpu` padding differs between model runners.** V2 fills empty slots with real tokens, which are proposed, included in the target forward pass and can be accepted. V1 pads with `-1` and trims those slots before the forward pass.
- **All draft tokens grammar-invalid (sync).** No draft tokens are scheduled, but the drafter did propose them. The step now counts as a spec step with 0 accepted. This slightly lowers the mean acceptance length on such workloads. #56977 considers counting such steps a bug. We count them on purpose, because the drafter should be penalized for proposing grammar-invalid draft tokens.
- **Drafter skipped.** When the longest sequence in the batch no longer fits the drafter's `max_model_len`, the drafter is skipped for the whole batch, and the V1 model runner passes zeros as draft tokens. The zeros are included in the target forward pass, so they count as verified, since this is real work. They also count as drafted, which keeps `drafted >= verified`. The acceptance rate drops to about 0 on those steps (#34734, #58080). We think the metrics are right to show this, and that the real fix is to not verify zeros at all (#34757, #52756). This differs from #34734, which asks to exclude these steps from the stats.
- **Truncated draft tokens (sync).** If the scheduler truncates the valid draft tokens (e.g. near `max_model_len`), the trailing invalid draft tokens are outside the scheduled window and are not counted.

</details>

<details>
<summary>C. Acceptance rate: accepted / drafted vs accepted / verified</summary>

`draft_acceptance_rate` is `accepted / drafted` today. Arguments for keeping it:

- **It does not depend on the scheduling mode.** `accepted / verified` would make async look worse than sync for the same draft tokens, only because async included the invalid draft tokens in the target forward pass.
- **It measures the drafter.** A drafter that proposes grammar-invalid tokens should be penalized for it.
- **One denominator everywhere.** Prometheus, logs, benchmarks and the per-request payload agree.

`accepted / verified` is still useful. It measures how well target compute was used, which is what AV tunes. Since verified is now reported, clients can compute it. (open question: should we expose it as its own field?)

</details>

<details>
<summary>D. Related issues and PRs</summary>

Open items that touch the same counts. #29821 (merged) is listed for context, because it introduced the subtraction of grammar-invalid draft tokens.

**Drafter skipped**

- #34734 (issue): stats still report drafted tokens when drafting is skipped, so the acceptance rate shows 0 beyond the drafter's context length.
- #34757 (PR, stale): signals the skip from the model runner and excludes those steps from the stats. Covers sync scheduling only.
- #52756 (issue): with async scheduling, draft tokens are still scheduled and verified after the drafter is skipped.
- #58080 (issue): MTP drafter misses `--hf-overrides`, so the length gate plus dummy-0 draft tokens silently kill acceptance on long contexts.

**Grammar-invalid draft tokens**

- #56977 (issue): `num_drafts` counts requests whose draft tokens were all invalidated.

**EOS and terminal steps**

- #56101 (issue): mean acceptance length overcounts when an EOS falls inside the accepted draft tokens.
- #56137 (PR): keeps raw accepted counts and reports committed tokens for mean acceptance length. Touches the same scheduler, stats, Rust and docs code as this proposal.
- #56195 (PR): competing fix that tracks emitted tokens.

**Variable draft length and acceptance rate**

- #58340 (PR): exposes the verified draft token count for AV. The implementation of this RFC builds on it.
- #56278 (PR): per-request acceptance histogram by draft length.
- #44487 (PR): Prometheus histogram of per-request acceptance rate.
- #61084 (PR): warns on sustained near-zero acceptance. Would fire on dummy-0 draft tokens.

</details>
