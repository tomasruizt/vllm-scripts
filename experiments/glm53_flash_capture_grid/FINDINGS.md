# GLM-5.3-Flash capture-grid pilot

Adding eight CUDA graph sizes improved throughput by 55% at C64 and 42% at C128. The KV capacity cost was 7.02%, substantially larger than the Qwen result.

| Concurrency | Default tok/s (range) | Up to 4K tok/s (range) | Gain | Valid repeats per variant |
|---|---:|---:|---:|---:|
| 64 | 3,326 (3,197–3,457) | 5,147 (5,124–5,174) | +54.77% | 3 |
| 128 | 4,963 (4,786–5,140) | 7,046 (7,035–7,058) | +41.97% | 2 |

All 12 measurements completed, with 3,764 requests and no errors, cancellations or preemptions. The first C128 measurement in each variant failed the saturation and minimum 10-second steady-window checks. These two measurements are preserved in the dataset and excluded from the comparison. The remaining 10 passed every check. This is a pilot with one server startup per variant, ordered Default then Up to 4K.

| Cost | Default | Up to 4K |
|---|---:|---:|
| Capture sizes | 85 | 93 |
| KV capacity (tokens) | 23,626,749 | 21,967,776 |
| Final graph-capture memory reported by rank 0 (GiB) | 6.23 | 12.88 |
| Final graph-capture phase (seconds) | 38 | 43 |
| Total startup (seconds) | 403 | 393 |

Up to 4K lost 7.02% of KV capacity and added five seconds to the final capture phase. Total startup was ten seconds shorter, but a single startup per variant cannot establish a startup improvement. The graph memory numbers describe the final capture phase, rather than all profiling and warmup work.

## Configuration

- Four B300s, TP4, native FP8 weights, FP8 KV, MTP5, default model runner and FlashKDA selection.
- vLLM `58b3298457dde7b4554b3b4e20b238c0ac2c3a65` (2026-10-02), matching the Qwen study.
- `zai-org/GLM-5.3-Flash`, revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`, all 62 shards reused from the shared cache.
- Resolved recipe defaults: context 1,048,576, token budget 16,384, maximum sequences 1,024, prefix caching enabled and reset before every measurement.
- AIPerf, SPEED throughput 1K, shuffled with seed 42, temperature zero, reasoning effort low, maximum completion 1,024, 30-second submission and 120-second grace period per measurement.
- Up to 4K preserves every default capture size and adds 1280, 1536, 1792, 2048, 2560, 3072, 3584 and 4096. Saved commands were compared and differ only in the capture configuration.

```bash
vllm serve zai-org/GLM-5.3-Flash \
  --revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a \
  --served-model-name glm53-flash --tensor-parallel-size 4 --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47 \
  --disable-uvicorn-access-log --cudagraph-metrics --host 127.0.0.1 --port 8778
```

The expanded variant appends `--compilation-config` using the exact JSON saved in `results/glm53-flash-capture-pilot-20261019/expanded-grid.json`. The runner uses the cached snapshot path in place of the Hub model ID and revision above.

```bash
aiperf profile --config configs/default/r2-c128.yaml
aiperf profile --config configs/4k/r2-c128.yaml
```

The runner generates all 12 configurations from `benchmark.yaml`. Rebuild the long-format dataset and graph coverage with `make fetch-results summarize`. Canonical data: `results/glm53-flash-capture-pilot-20261019/metrics.parquet`, including all original AIPerf summaries and per-request records.

## Interpretation

The benefit extends to GLM-5.3-Flash on this workload. This supports testing larger capture defaults beyond Qwen, while measuring the model-specific memory cost. It does not establish the best grid or performance at other concurrency levels and prompt lengths.

Graph coverage is derived from native dispatch histograms in complete logging intervals inside the valid client runs. It measures steps dispatched as FULL or PIECEWISE graphs, rather than the fraction of requests or GPU execution time. KDA retains eager regions inside PIECEWISE execution. Steps above `6 × concurrency` must include prefill tokens with MTP5, but smaller steps can also contain prefill tokens.

| Concurrency | Default graph coverage, all steps | Up to 4K, all steps | Default, above decode-only bound | Up to 4K, above decode-only bound |
|---|---:|---:|---:|---:|
| 64 | 77.15% | 100.00% | 0.00% | 100.00% |
| 128 | 75.96% | 99.72% | 0.00% | 99.02% |
