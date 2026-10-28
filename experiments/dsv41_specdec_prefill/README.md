# DSV4.1-Flash: prefill cost of speculative decoding

Reproduces Robert's random-prompt benchmark (8,192 input tokens, 16 output tokens, prefix caching off, c=1 and c=16) on 8×H100, with AIPerf instead of the original client.
It compares no speculation, dspark K=3, K=5 and K=5 with adaptive verification.
It also answers whether the drafter turns off DSV4.1's decoder SWA bounded replay (the YOCO-style prefill shortcut), from each server's startup log.

## Setup

* Node: RH-h100-3 (8× H100 80GB HBM3), each variant under `chg run --gpus 8`, one variant after another.
* Model: `deepseek-ai/DeepSeek-V4.1-Flash-1017` (FP8 with FP4 experts) from the shared HF cache on `/mnt/nfs-preprod-1`.
* vLLM: main at 71e9320633 (2026-10-22), precompiled wheels.
* Server flags: the production config from the IBM k3/k8a campaign, plus `max_cudagraph_capture_size` 4096 (as in the adaptive verification blog post) and `--no-enable-prefix-caching`.
  Cluster-only flags are dropped: KV events, `OffloadingConnector` CPU offload, `--enable-scale-out` and the shared-memory multimodal cache.
  `--reasoning-effort-map` is dropped too, because upstream vLLM does not have it (the workload sends `reasoning_effort: none` anyway).
* `--engram-config '{"cpu_offload":true}'` is kept, to match production and to save GPU memory (the 510 GB checkpoint leaves little room on 8×H100).
  This model's Engram layers (1 and 14) sit before the last KV-source layer (20), so Engram cannot turn off bounded replay.
* Speculative decoding uses `draft_sample_method: probabilistic`, as in production.
* Workload: AIPerf `--prompt-corpus random --random-corpus-style vllm` (random token ids, like `vllm bench serve --dataset-name random`), `ignore_eos`, thinking off.
  Each concurrency runs 3 trials of 60 s after a short warmup, with a new seed per trial.

## Run

```bash
make launch   # copy scripts to the node and start in the background
make watch    # wait for completion, then fetch-results and summarize into results/<run>/metrics.parquet
```
