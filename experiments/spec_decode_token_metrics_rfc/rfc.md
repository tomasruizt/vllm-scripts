# [RFC]: Consistent semantics for drafted, verified and accepted spec decode tokens

## Summary

vLLM reports how many draft tokens were drafted and accepted during speculative decoding (SD). With adaptive verification (AV) and structured outputs (SO), it is no longer clear what `drafted` means. We propose three clear definitions, one invariant that always holds, and to report all three metrics when using SD.

| Count | Meaning |
| --- | --- |
| **Drafted** | Draft tokens proposed by the drafter. |
| **Verified** | Draft tokens the target model ran a forward pass on. |
| **Accepted** | Draft tokens the target model accepted. |

Invariant: `drafted >= verified >= accepted`, in total and at every step.

This RFC grew out of reviewing #58340, and discussions with @benchislett and @yzong-rh.

## Background

Two features make `drafted` and `verified` differ.

**Adaptive verification.** The drafter proposes `k` tokens, but the target only verifies a prefix of them, chosen by a confidence score. This saves target compute when the later draft tokens are unlikely to be accepted. AV is only available in the V2 model runner (MRV2), not in the V1 model runner (MRV1). The skipped draft tokens were proposed, but never verified.

**Structured outputs.** A grammar (e.g. a JSON schema) constrains the output. A draft token that violates the grammar can never be accepted. When vLLM checks the grammar depends on the scheduling mode:

- **Sync scheduling** checks the draft tokens against the grammar before the forward pass. Invalid draft tokens are dropped and never included in the target forward pass.
- **Async scheduling** launches the forward pass first and checks the grammar on the CPU in parallel, to keep the GPU busy. Invalid draft tokens are included in the target forward pass, then masked out before sampling, so they are always rejected.

## Problem

Today, `num_draft_tokens` counts the draft tokens that passed the grammar check. This is neither the number of proposed tokens nor the number of verified tokens:

- **It is not "proposed tokens".** Grammar-invalid draft tokens are subtracted, but draft tokens skipped by AV are kept.
- **It is not "verified tokens".** Async scheduling subtracts grammar-invalid draft tokens, even though they were included in the target forward pass.

On top of that, `verified` is not reported. With AV, there is no way to see how many draft tokens the target actually verified.

If we add a `verified` count without changing `drafted`, the invariant can be violated. This happens when async scheduling, SO and AV are used together.

Sync and async scheduling also count spec steps differently. If all draft tokens of a step are grammar-invalid, async counts the step and sync does not (#56977). This changes the mean acceptance length.

## Proposal

### 1. Drafted means proposed by the drafter

`num_draft_tokens` counts every token the drafter proposed, including:

- draft tokens that violate the grammar (both scheduling modes),
- draft tokens skipped by AV.

Padding slots (e.g. from `ngram_gpu`) count only if the target runs a forward pass on them, see [appendix B](#appendix).

### 2. Verified means included in the target forward pass

`num_verified_tokens` counts draft tokens the target model ran a forward pass on. It measures how much target compute went into verification.

- With AV, the model runner reports it, because only the model runner knows which draft tokens it verified.
- Otherwise, the scheduler computes it as `min(scheduled draft tokens, proposed draft tokens)`. This counts grammar-invalid draft tokens in async scheduling (included in the target forward pass), and drops them in sync scheduling (not included).

### 3. Count spec steps the same way in both modes

A step counts as a spec step (`num_drafts`, `num_spec_steps`) if the drafter proposed tokens for it, even if all of them are grammar-invalid. The prefill step still does not count. The mean acceptance length stays `1 + accepted / spec steps`.

### 4. Report all three metrics everywhere

`verified` is reported wherever `drafted` and `accepted` are: Prometheus, the periodic log line, the per-request metrics, `vllm bench serve` and the Rust frontend. See [appendix A](#appendix) for the field names.

A client should use `drafted` to judge the drafter, `verified` to measure target compute, and the mean acceptance length to estimate the speedup.

## Worked examples

All examples use `k = 4`. The last column shows today's `drafted` count.

| Case | What happens | `drafted` | `verified` | `accepted` | `drafted` today |
| --- | --- | --- | --- | --- | --- |
| No grammar, no AV | All 4 draft tokens in the target forward pass, target accepts 2 | 4 | 4 | 2 | 4 |
| AV | Drafter proposes 4, target verifies the first 3, accepts 2 | 4 | 3 | 2 | 4 |
| Sync scheduling, grammar rejects last 2 draft tokens | Only 2 draft tokens in the target forward pass, target accepts 1 | 4 | 2 | 1 | 2 |
| Async scheduling, grammar rejects last 2 draft tokens | All 4 in the target forward pass, last 2 masked, target accepts 1 | 4 | 4 | 1 | 2 |
| Async + AV, grammar rejects last 2 draft tokens | Target verifies the first 3 (the 3rd is invalid), accepts 1 | 4 | 3 | 1 | 2 (less than `verified`) |
| Sync scheduling, grammar rejects all 4 draft tokens | No draft tokens in the target forward pass | 4 | 0 | 0 | 0 (step skipped) |

The acceptance rate (`accepted / drafted`) is the same for sync and async scheduling (1/4). Only `verified` differs, which reflects the real difference in compute.

See [appendix B](#appendix) for the edge cases behind these rows, e.g. how async draft tokens enter the forward pass before the grammar check, and how padding slots are counted.

## Impact on users and clients

Without SO, `drafted` and `accepted` do not change, and `verified` equals `drafted` unless AV is on.

With SO:

- `num_draft_tokens` goes up, and `draft_acceptance_rate` goes down. The new numbers are the same in sync and async scheduling.
- The mean acceptance length only changes in sync scheduling, when all draft tokens of a step are grammar-invalid.
- Clients that compute `accepted / draft_tokens` from Prometheus will see lower rates. This includes AIPerf's speed-bench report, the Datadog integration and many user scripts.
- Clients that use `drafted` to estimate target compute should use `verified` instead.
- AIPerf's per-request adapter keeps working. It ignores new fields, and all of its consistency checks still hold. Its docs say vLLM subtracts grammar-invalid draft tokens, which would need an update.
- The new Prometheus counter name contains neither `num_draft_tokens` nor `num_drafts`, so parsers that match metric names by substring (like `vllm bench serve`) do not double count.

There is no flag for the old behavior. We will note the change in the release notes and update the docs.

`verified` is always reported, even when it equals `drafted`, so the output format does not depend on the config. This adds two Prometheus counters (one of them per position).

## Alternatives considered

- **`drafted` means "sent to verification".** This would not penalize the drafter for grammar-invalid tokens, although a better drafter would have proposed valid ones. It also clashes with the domain language: AV skips draft tokens without verifying them, but "drafted but not verified" would become a contradiction.
- **Rename `drafted` to `proposed`.** The two words mean the same thing, and a rename breaks existing dashboards and clients. It also does not solve the problem, since we would still need `verified`.
- **Only add `verified`.** The smallest change, but `verified` can then be larger than `drafted`, violating the invariant (see [Problem](#problem)).

## Out of scope

- **Draft tokens after a stop.** Stats are recorded before the output is cut at the stop (EOS, stop token ids, `max_tokens`). So `accepted` can include tokens after the stop, which are discarded. This is tracked in #56101, with two proposed fixes: clip `accepted` at the stop, or report committed tokens separately (#56137, #56195). Both keep the invariant.

## Open questions

1. Is `accepted / drafted` the right default for `draft_acceptance_rate`, or should we also expose `accepted / verified` as its own field? See [appendix C](#appendix) for arguments.
2. Should draft tokens skipped by AV lower the acceptance rate? They were not rejected by the target, but a better drafter could have produced higher-confidence draft tokens. As proposed, they count in `drafted`, so they do lower it.

## Appendix

<details>
<summary>A. Where each count is reported</summary>

| Surface | `drafted` | `verified` | `accepted` |
| --- | --- | --- | --- |
| Prometheus | `vllm:spec_decode_num_draft_tokens` | `vllm:spec_decode_num_verified_tokens` (new) | `vllm:spec_decode_num_accepted_tokens` |
| Periodic log line | `Drafted:` | `Verified:` (new) | `Accepted:` |
| Per-request metrics | `num_draft_tokens`, `per_step_drafted` | `num_verified_tokens`, `per_step_verified` (new) | `num_accepted_draft_tokens`, `per_step_accepted` |
| `vllm bench serve` | yes | new | yes |
| Rust frontend | yes | new | yes |

</details>

<details>
<summary>B. Edge cases in how draft tokens reach the target forward pass</summary>

- **Async draft tokens enter the target forward pass before the grammar check.** The forward pass for step N starts while the CPU still validates the draft tokens against the grammar. So invalid draft tokens cost target compute, even though they can never be accepted. This is why they count as `verified` on that path.
- **Padding slots count only if the target runs a forward pass on them.** When `ngram_gpu` finds fewer than `k` matches, MRV2 fills the empty slots with the last sampled token and verifies them, so they count. MRV1 pads with `-1` and removes those slots before the forward pass, so they don't. Exception (pre-existing): with MRV1 and without SO, the scheduler doesn't see the removed slots and still counts all `k`.
- **All draft tokens grammar-invalid (sync).** No draft tokens are scheduled, but the drafter did propose them. The step now counts as a spec step with 0 accepted. #56977 considers counting such steps a bug. We count them on purpose, because the drafter should be penalized for proposing grammar-invalid draft tokens.
- **Drafter skipped.** When the longest sequence in the batch no longer fits the drafter's `max_model_len`, the drafter is skipped for the whole batch, and MRV1 passes zeros as draft tokens. The zeros are included in the target forward pass, so they count as `verified`, since this is real work. They also count as `drafted`, which keeps `drafted >= verified`. The acceptance rate drops to about 0 on those steps (#34734, #58080). We think the metrics are right to show this, and that the real fix is to not verify zeros at all (#34757, #52756). This differs from #34734, which asks to exclude these steps from the stats.
- **Truncated draft tokens (sync).** If the scheduler truncates the valid draft tokens (e.g. near `max_model_len`), the trailing invalid draft tokens are not counted in `drafted`. This is the one exception to Proposal 1: `drafted` only counts draft tokens up to the scheduler's limit.

</details>

<details>
<summary>C. Acceptance rate: <code>accepted / drafted</code> vs <code>accepted / verified</code></summary>

`draft_acceptance_rate` is `accepted / drafted` today. Arguments for keeping it:

- **It does not depend on the scheduling mode.** `accepted / verified` would make async look worse than sync for the same draft tokens, only because async included the invalid draft tokens in the target forward pass.
- **It measures the drafter.** A drafter that proposes grammar-invalid tokens should be penalized for it.
- **One denominator everywhere.** Prometheus, logs, benchmarks and the per-request payload agree.

`accepted / verified` is still useful. It measures how well target compute was used, which is what AV tunes. Since `verified` is now reported, clients can compute it. (open question: should we expose it as its own field?)

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
