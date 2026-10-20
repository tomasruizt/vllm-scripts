# Repeated Qwen startup measurements

We completed two additional startups per capture grid, with one full serving sweep after each startup. Together with the original experiment, all 90 client measurements across nine server startups passed the validity checks.

| Variant | Actual graph mem | Estimated graph mem | KV cache mem | Token capacity | Capture time range | Startup time range |
|---|---:|---:|---:|---:|---:|---:|
| Default | 2.85 GiB/GPU | 4.28 GiB/GPU | 198.14 GiB/GPU | 26.64M | 35–38 s | 217–245 s |
| Up to 2K | 2.68 GiB/GPU | 3.96 GiB/GPU | 198.45 GiB/GPU | 26.68M | 36–37 s | 202–250 s |
| Up to 4K | 4.04 GiB/GPU | 5.09 GiB/GPU | 197.32 GiB/GPU | 26.53M | 38–39 s | 207–229 s |

The memory figures repeat exactly at the reported precision across all three startups per variant, including the lower allocation for Up to 2K. KV token counts are also identical across startups. Calling this startup noise was not supported. The allocation-level cause remains open under Beads rh-50y. Capture and total startup times do vary.

Additional runs used the same pinned vLLM, checkpoint, server arguments and client workloads. Two four-GPU groups ran concurrently. Each variant used both groups across the additional startups. The independent `startup` column and client `repeat` column are retained separately in the combined Parquet file.

```bash
make fetch-results RUN_ID=qwen38-recipe-restarts-20261020
make summarize-startups
```

Canonical data: `results/qwen38-recipe-restarts-20261020/metrics.parquet`. The same directory contains graph-shape samples, all-step and prefill/mixed coverage CSVs, and updated plots pooled across all 90 client runs. The local issue draft includes these results.
