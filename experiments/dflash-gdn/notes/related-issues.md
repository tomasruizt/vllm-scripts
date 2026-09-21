The closest reports I found are:
- #50722: poor Qwen3.5 DFlash performance despite acceptance lengths around 5–6. It uses the 35B-A3B MoE model on A100, so it’s a similar symptom under different conditions.
- #42505: weak DFlash speedup at concurrency 1 and slowdowns above concurrency 8.
- PR #43594: reduces DFlash’s per-layer KV-cache write launches from eight to one—a relevant source of host launch overhead.
- #51008: discusses GDN host overhead, but the author retracted the claim that it affects every speculative step. It doesn’t establish the cause of our gap.