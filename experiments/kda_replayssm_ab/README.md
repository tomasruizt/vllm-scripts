# KDA ReplaySSM E2E A/B on GLM-5.3-Flash (spec)

What to measure and how to judge it. Runner scripts are left to the implementer. `benchmark.yaml` is the AIPerf config (placeholders filled per cell), `prepare_data.py` builds the dataset.

## Variants

`{main,pr} x AV {off,on}`. `main` is the branch's merge base. 4 B300 per server.

```bash
vllm serve zai-org/GLM-5.3-Flash --revision 46a050bec13002ad29ee795c16e61cf4fa3c4a9c \
  --tensor-parallel-size 4 --kv-cache-dtype fp8 --max-model-len 8192 --max-num-seqs 256 \
  --max-cudagraph-capture-size 2304 --no-enable-prefix-caching --reasoning-parser glm47 \
  --speculative-config '{"method": "dspark", "model": "RedHatAI/GLM-5.3-Flash-speculator.dspark", "revision": "738c283a7e860d7dafce5785161466bc556e5cae", "num_speculative_tokens": 8, "enable_adaptive_verification": <false|true>}' \
  [--use-replayssm]   # pr only
```

K=8 is the speculator's block size (K=7 only if the kernel caps at 8 tokens). Capture size 256 x (K+1) keeps c256 decode in CUDA graphs.

## Workload and schedule

* SPEED-Bench `throughput_1k` (HF revision `fbd6dc64`), resolved with NVIDIA's Model-Optimizer `specdec_bench` loader (`509acae9`), since 639 of 1536 rows on HF are placeholders. AIPerf 0.13.0, temperature 1.0, top-p 0.95, max 1024 output tokens.
* Cells: 30 s at c<=32, +10 s per doubling above, 120 s grace.
* 2 sessions per variant. Per session: warmup cell (discarded), 3 repeats of c1, 4, 16, 64, 128, 256, then GSM8K.
* Interleave main and pr: with 8 GPUs in parallel on two groups, swapping groups in the second round, with 4 GPUs in ABBA order. Record host CPU load and GPU clocks, and keep the host otherwise idle.
## Analysis

* Throughput: steady-window output tok/s (from the c-th completion to the last submission, `benchmarking/aiperf.py:steady_window`) and TPOT. Drop cells with errors, preemptions, queueing, unsaturated batches or CPU contention (`gates`).
* Gain per concurrency and AV setting: median pr vs median main, also per session pair. It counts only if both pairs agree in sign.

## Correctness (pass if the 95% CI of pr - main contains 0)

* GSM8K (5-shot, same sampling, seeds shared between main and pr), paired by question over all runs. Half-width ~0.8 pp, so a 2 pp drop fails. The eval script needs a patch to save per-question correctness.
* Acceptance length with AV off only, since AV adapts the verify budget to the forward cost, which ReplaySSM changes.

## Memory

Expected savings: 307 to 407 MiB per GPU per request, i.e. 77 to 102 GiB per GPU at c256, and ~3.5x max concurrency at 8K tokens.
Without prefix caching, main keeps 1+K = 9 state slots per request in each of the 34 KDA layers, ReplaySSM 1. A slot is 1.13 MiB per GPU (fp32 state 16 x 128 x 128, plus conv state), so 8 x 34 x 1.13 = 307 MiB. Allocator padding can raise this to 407 MiB (KDA grouped by the 11 attention layers into 44 slots, pages rounded up to the attention block). An in-pool replay ring (25 tokens of k, v, g, beta per layer) gives back ~15 MiB.

Measure: freed = main - pr of KV pool / max concurrency (startup log). Compare the pool sizes too, since buffers outside the pool (like the Kimi-K3 RecoverSSM records) shrink it. Also report `vllm:kv_cache_usage_perc` per cell.
