# DSV4.1-Flash: end-to-end validation of PR #59986

PR #59986 downgrades five DSV4.1 sparse metadata builders from `AttentionCGSupport.ALWAYS` to `UNIFORM_BATCH`, and reports `1 + num_speculative_tokens` as their varlen decode bound.
With `ALWAYS`, the cudagraph resolver lets batches with a prefill replay FULL graphs that were captured on all-decode batches, although the builders split decodes from prefills on the host.
This experiment checks both reported symptoms on base vs PR, and measures the cost of keeping mixed batches out of FULL graphs.

* #62417: DSpark chat/thinking output degenerates into loops with `cudagraph_mode=FULL`.
* #59970: illegal memory access with DSpark adaptive verification, `FULL_AND_PIECEWISE` and concurrent prefill.

## Setup

* Node: RH-nm-ore-b300-01 (B300, SM103), each variant under `chg run --gpus 2` (TP2 + EP).
* Model: `deepseek-ai/DeepSeek-V4.1-Flash-1017` from the node's shared HF cache.
* vLLM: base is main at 71e9320633 (2026-10-23), PR is the same commit plus the #59986 commit d29af5185c (cherry-picked, uncommitted). Both share one precompiled build, since the PR is Python-only.
* Attention: `FLASHINFER_MLA_SPARSE_DSV41`, fp8 KV and indexer as in both issues, block size 128 (the only size this backend supports on SM103, #62417 used 64 on SM120 with a local patch).

| Variant | Tree | cudagraph mode | Speculation | Phases |
|---|---|---|---|---|
| `base-full`, `pr-full` | base, PR | `FULL` | dspark k=3 | probes, GSM8K, perf |
| `eager` | base | `--enforce-eager` | dspark k=3 | probes, GSM8K (reference) |
| `base-av`, `pr-av` | base, PR | `FULL_AND_PIECEWISE` | dspark k=5, adaptive verification, plus the #59970 flags (mxfp4 sparse-logit indexer, prefix caching, long prefill threshold 4096, no FlashInfer autotune) | 10 min load at c96, then probes |

* Probes: the #62417 prompts (short math, one-word answer, Chinese, tool call, multi-turn chat), greedy, thinking on, one request at a time.
* GSM8K: vLLM's `tests/evals/gsm8k/gsm8k_eval.py` through chat completions, 200 questions, greedy, `max_tokens` 8192.
* Perf: AIPerf random prompts, ISL 1024, OSL 256, `ignore_eos`, thinking off, c1/8/32/64 for 45 s each.
* Load: AIPerf closed-loop multi-turn chat, 2000 conversations of about 6 turns, 600-token shared system prompt, turn ISL 2048±2048, OSL 600±800, `ignore_eos`, c96 for 10 min, so prefix-cached prefills keep joining decode batches.
  Run it on its own with `make launch RUN_ID=<new id> LAUNCH_ENV='ROUNDS=base-av,pr-av'`.

## Run

```bash
make setup    # create the base and PR source trees on the node
make launch   # copy scripts to the node and start in the background
make watch    # wait for completion, then fetch-results and summarize into results/<run>/metrics.parquet
```

| Branch | main | main | PR #59986 |
|---|---|---|---|
| Server arg | `--enforce-eager` | `--compilation-config '{"cudagraph_mode": "FULL"}'` | `--compilation-config '{"cudagraph_mode": "FULL"}'` |
| Resolved cudagraph mode | none (eager) | FULL | FULL_DECODE_ONLY |
| GSM8K accuracy | ✅ 0.955 | ❌ 0.020 | ✅ 0.955 |
| GSM8K invalid | ✅ 0.000 | ❌ 0.980 | ✅ 0.005 |
| `17*19=?` probe ([#62417](https://github.com/vllm-project/vllm/issues/62417)) | ✅ `323` | ❌ no answer, garbage text up to the 2048-token limit | ✅ `323` |
| Tool-call probe ([#62417](https://github.com/vllm-project/vllm/issues/62417)) | ✅ successful tool call | ❌ garbage text, no tool call | ✅ successful tool call |
| Illegal memory access ([#59970](https://github.com/vllm-project/vllm/issues/59970)) | ✅ no | ❌ yes (in the perf sweep at c1) | ✅ no |
