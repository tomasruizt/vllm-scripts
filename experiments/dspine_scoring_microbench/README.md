# DSpine transition scoring microbenchmark

Times DSpine's `_score_transitions` from vLLM PR #67094 (head `c1ef7580c6`, 2026-11-11) at `tencent/Qwen3.8-27B-DSpine-b16` shapes (H=5120, transfer 1024, message 512, top-k 16, 15 draft tokens, V=248320), with random weights, inside CUDA graphs. The batch size is the number of requests, each scoring 15 positions.

Variants:
- `ref`: PR code, verbatim.
- `ref_compiled`: `torch.compile` of `ref`.
- `lowrank`: the same math in the 512-dim message space, with per-vocab tables.
- `fused`: `lowrank` with a Triton per-position kernel and a TF32 `u^T G u` mm.
- `lmhead_topk`: the draft's full-vocab lm_head + top-k, as a yardstick.

## Results (H100)

| batch | `ref` | `fused` |
|---|---|---|
| 8 | 0.82 ms | 0.15 ms |
| 256 | 19 ms | 1.6 ms |

`fused` matches `ref` within 5e-6 (max abs score error). The vLLM integration lives on branch `perf/dspine-scoring` of tomasruizt/vllm (`VLLM_DSPINE_SCORING=fused`), E2E numbers in [../dspine_e2e](../dspine_e2e).

```bash
make launch   # copies bench.py to the remote and runs it under `chg run`
make watch    # waits for it, then fetches results/<RUN_ID>/
```
