# [Tracking]: Hidden-state extraction for GLM-5.3-Flash

## TLDR

| PR | Status | Why |
|---|---|---|
| [#56983](https://github.com/vllm-project/vllm/issues/56983) | In review | Aux hidden-state capture in `glm5next` |
| [#57169](https://github.com/vllm-project/vllm/issues/57169) | In review | Replace GLM-5.3-Flash's custom kv-layout with the generic packed one |
| none yet | Needs testing on top of #57169 | `HiddenStateCacheSpec` + connector under `BLHNC` |

## Motivation

We want to use `method: extract_hidden_states` on `zai-org/GLM-5.3-Flash`, for example to generate training data for EAGLE3/DFlash/DSpark drafters with speculators. On main (`25b0add7b8`) this doesn't work. Several PRs cover parts of it, mostly aimed at DFlash2 serving. This issue lists what has to be solved and which PRs and issues address each part.

## What is missing on main

1. `glm5next` does not implement `SupportsEagle3` / `EagleModelMixin`, so there is no aux hidden-state capture. Startup fails with `Model does not support EAGLE3 interface`.
2. From reading the code on main, the KV cache groups can't be built once a `HiddenStateCacheSpec` layer is present:
   - `_get_kv_cache_groups_glm5_next` requires `type(spec) is MLAAttentionSpec`. `HiddenStateCacheSpec` is a subclass of `MLAAttentionSpec`, so the check fails and the function returns `None`.
   - The generic fallback then fails on page-size unification for `indexer.k_cache`, because MLA pages can't be padded.
   - The full-allocation fallback explicitly rejects `HiddenStateCacheSpec`.

## What needs to be solved

### 1. Aux hidden-state capture in `glm5next`

- `glm5next` must implement `SupportsEagle3` and capture the mHC-complete layer output (`hc_post` + `hc_contract`, not `hidden + residual`). [#56983](https://github.com/vllm-project/vllm/issues/56983) does this (reviewer: @ivanium). [#58834](https://github.com/vllm-project/vllm/issues/58834) fixes the sequence-parallel layout of the non-mHC layers in the same file, which the aux capture relies on.
- **Recommendation:** land [#56983](https://github.com/vllm-project/vllm/issues/56983).
- Related: [#55682](https://github.com/vllm-project/vllm/issues/55682) (same capture, with unit tests; largely redundant), [#55423](https://github.com/vllm-project/vllm/issues/55423) and [#55620](https://github.com/vllm-project/vllm/issues/55620) (earlier attempts), [#54451](https://github.com/vllm-project/vllm/issues/54451) (issue).

### 2. Generic kv-layout for GLM-5.3-Flash

At startup, GLM-5.3-Flash's custom grouping (`_get_kv_cache_groups_glm5_next`) doesn't support `HiddenStateCacheSpec`: it only knows GLM-5.3-Flash's own cache types (MLA, indexer, kpool tail, KDA state) and has no place for an extra layer in its kv-layout. vLLM then falls back to the generic grouping, which can't unify GLM-5.3-Flash's MLA and indexer page sizes, and startup fails with `NotImplementedError` on `indexer.k_cache`.

- **Generic packed kv-layout (recommended):** [#55219](https://github.com/vllm-project/vllm/issues/55219) / [#57169](https://github.com/vllm-project/vllm/issues/57169) remove the custom kv-layout and route GLM-5.3-Flash through the generic packed (`BLHNC`) kv-layout, which supports mixed page sizes. They are not aimed at hidden-state extraction. Land [#57169](https://github.com/vllm-project/vllm/issues/57169).
- **Extend the custom kv-layout:** the [`shanjiaz/vllm` fork](https://github.com/shanjiaz/vllm/tree/glm5-next-aux-hidden-state-concurrency-fix) appends the hidden-state group and places its storage after the native pages. It works under concurrent load, but the custom GLM-5.3-Flash kv-layout code is being replaced by [#57169](https://github.com/vllm-project/vllm/issues/57169).

### 3. `HiddenStateCacheSpec` and the extractor/connector under `BLHNC`

The generic packed path already adds hidden-state layers as their own KV cache group, so it may work as is on top of [#57169](https://github.com/vllm-project/vllm/issues/57169), but this hasn't been tested with GLM-5.3-Flash. To check:

- **Block size:** the hidden-state group keeps GLM-5.3-Flash's block size (1152 tokens on GB200 TP4), so its page would be ~66 MB (~57 KB per token). The packed kv-layout sizes every block by the largest group's page, so the native blocks (~13 MB) would grow to ~66 MB and be ~80% padding, cutting the target's KV capacity ~5×. This is a memory cost, not a correctness issue; if confirmed, a smaller hidden-state block size (e.g. 128–256 tokens) would fix it in a small follow-up PR.
- **Connector:** `ExampleHiddenStatesConnector` asks for `LBNHC`. The resolver falls back to `BLHNC` with a warning, but reading and writing hidden states under `BLHNC` is untested.
- **Tests:** a regression test that no hidden-state page overlaps a native page, and an end-to-end test with concurrent, mixed-length requests that checks each request's hidden states are finite, have the expected shape, and agree with a serial reference.

Related: [#56822](https://github.com/vllm-project/vllm/issues/56822) (hidden-state memory accounting on hybrid models), [#53074](https://github.com/vllm-project/vllm/issues/53074) (DeepSeek-V4), [#50894](https://github.com/vllm-project/vllm/issues/50894) (TP page size).

Keep an eye on:
- [#53558](https://github.com/vllm-project/vllm/issues/53558): pluggable `KVCacheConfigBuilder` that lets models and platforms override KV cache planning; it also touches the hidden-state connector tests.
- [#58636](https://github.com/vllm-project/vllm/issues/58636) / [#58979](https://github.com/vllm-project/vllm/issues/58979): the GLM-5.x sparse indexer's key norm is autotuned per TP rank, so outputs can differ between launches. This affects comparisons against a serial reference in the end-to-end test.

## Adjacent (not required for extraction)

- [#58454](https://github.com/vllm-project/vllm/issues/58454): kpool corruption with speculative decoding (k ≥ 2).
- [#55800](https://github.com/vllm-project/vllm/issues/55800): sliding-window DFlash drafter admission scales with the full sequence length.
- [#58638](https://github.com/vllm-project/vllm/issues/58638): RFC on KV grouping for hybrid models with drafters.
- [#56417](https://github.com/vllm-project/vllm/issues/56417) / [#56442](https://github.com/vllm-project/vllm/issues/56442): `extract_hidden_states` uses the first column's feedback for multi-token outputs (only relevant with `max_tokens > 1`).

### Speculators-related

- **Stale captures with prefix caching:** the speculators `dynamic_hidden_states` plugin (`AuxLayerWorkerExtension`) lets you change the captured layer set at runtime through `POST /aux_hidden_state_layers`. The hidden-state cache blocks are prefix-cached like any other KV block, so after a layer switch a prefix-cache hit returns hidden states from the old layer set. Upstream vLLM fixes the layer set at startup, so this can't happen there. Workaround: `--no-enable-prefix-caching`.
- **torch.compile / CUDA graphs:** the plugin's compiled path for dynamic aux-layer selection doesn't cover GLM-5.3-Flash yet, so it has only been validated with `--enforce-eager`.
