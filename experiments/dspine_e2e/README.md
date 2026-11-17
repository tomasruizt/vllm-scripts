# DSpine E2E: step profiling and equivalence

Qwen3.8-27B + `tencent/Qwen3.8-27B-DSpine-b16` (vLLM PR #67094), offline engine on GSM8K prompts.

- [profile_steps.py](profile_steps.py): per-step GPU time (target, verify+sample, draft) with CUDA events around the model runner, per concurrency. Scoring variant via `VLLM_DSPINE_SCORING` (`ref`, `fused`, ...) from branch `perf/dspine-scoring` of tomasruizt/vllm.
- [launch_profile.sh](launch_profile.sh): runs A/B sequences of variants on the same GPU, e.g. `SEQUENCES="ref-k7,fused-k7;fused-k7,ref-k7"` (variant format `<scoring>[_tf32][-k<num_spec_tokens>]`).
- [launch_nsys.sh](launch_nsys.sh): nsys kernel breakdown of one variant.
- [equivalence.py](equivalence.py), [launch_equivalence.sh](launch_equivalence.sh): greedy outputs, acceptance length and accuracy of two builds, for refactors that must not change the computation.

```bash
make launch-profile RUN_ID=<id> SEQUENCES="ref,fused;fused,ref" PROFILE_ARGS="--concurrency 1 8 32"
make watch RUN_ID=<id>   # waits, then fetches results/<id>/
```

## Findings

Loading refactor (branch `dspine-refactor`) vs PR head on H100, 128 GSM8K prompts, greedy: identical tokens, acceptance length 7.19, accuracy 0.83.

B200, `RedHatAI/Qwen3.8-27B-FP8`:
- Concurrency cap: vLLM reserves `1 + num_speculative_tokens` fp32 GDN state blocks per request and mamba group (10 groups). At K=15 that is about 2.7 GiB of GDN state per request (16x more than without spec decode, independent of context length), capping the engine at about 34 concurrent requests. K=7 raises the cap to about 68 (4096-token requests).
- K=15 vs K=7 (`fused`): K=15 is faster per user at c1 (646 vs 544 tok/s/user) and c8. K=7 wins from c32 up and saturates at about 6450 vs 4050 tok/s (+58%). Target forward at c32: 43.6 ms (K=15) vs 20.8 ms (K=7).
- `fused` vs `ref` scoring at K=7: +1.6%, 1.8%, 2.4%, 3.1%, 3.0%, 2.9% throughput at c 1, 8, 32, 64, 128, 256. Draft step at the cap: 8.4 ms to 5.8 ms. At K=15: 1.7% (c1), 2.8% (c8), 3.5% (c32) step time saving.
- K=7 with the K=16 checkpoint needs the relaxed block-size check (WIP commit on `perf/dspine-scoring-v2`).
- `--use-replayssm` is rejected for `Qwen3_5ForConditionalGeneration` on main. RecoverSSM for GDN (vllm PR #67590) would remove the per-draft state copies.
