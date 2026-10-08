# GLM-5.3-Flash extraction results

**MRV2 now selects piecewise CUDA graphs automatically and passes extraction through Speculators' actual client.**
No explicit graph-mode override is needed with the fix.

| MRV2 configuration | Outcome |
|---|---|
| Before fix, default graph mode | 20 client requests passed, but graphs were disabled |
| Before fix, explicit `PIECEWISE` | 40 requests passed, including 20 exact-reference comparisons |
| With fix, default graph mode | 40 requests passed, including 20 exact-reference comparisons and verified graph replay |

The fix tells the graph-mode resolver that MRV2 has a breakable-graph implementation available.
It can then fall back to piecewise graphs when extraction's storage backend cannot use full graphs, instead of disabling graphs entirely.

In the latest run, every reference chunk replayed target-model graphs and every saved output matched its same-execution reference byte-for-byte.
Token coverage was complete, and the checker detected corrupted values, swapped layers, and shifted tokens.
The storage helper itself ran outside graphs.
No equality between separate batches was assumed.

Validation: 10 graph-mode resolver tests, Ruff lint/format, Python 3.12 mypy, and the full-model API run passed.
The earlier 136 cache/connector unit tests also passed.
Scope remains text prompts, layers 5/22/43/45, TP4, and prefix caching disabled.

Evidence: [request results](/home/tomasruizt/benchmarks/glm-hs-validation/api-mrv2-default-fixed/results.json), [replay counts](/home/tomasruizt/benchmarks/glm-hs-validation/api-mrv2-default-fixed/same_replay_reference-graphs.json), [server log](/home/tomasruizt/benchmarks/glm-hs-validation/api-mrv2-default-fixed/server.log), and the exact [tested patch](/home/tomasruizt/benchmarks/glm-hs-validation/api-mrv2-default-fixed/vllm.patch).
See [VALIDATION.md](VALIDATION.md) for reproduction and limits.

Setup: vLLM `f95f7f102` (the tested patch on `7ca299b9c`), Speculators `e3041ddb`, four B300 GPUs, and the shared FP8 snapshot `eb9eb208` on `/data`.
No weights were downloaded or copied.
The fix is committed and pushed to PR #59037; its description includes these results.
