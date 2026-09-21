# Qwen3.5-4B DFlash reproduction and SGLang comparison

Issue: [vllm-project/vllm#49730](https://github.com/vllm-project/vllm/issues/49730). Runs used one NVIDIA H100 80GB HBM3 reserved with `canhazgpu`, TP 1, FP8 target weights, Qwen/Qwen3.5-4B, z-lab/Qwen3.5-4B-DFlash, streamed chat completions, GSM8K questions, 256 output tokens, 20 warmups, and concurrency 1. The primary comparison used the same ordered input payloads for all four configurations, with 200 measured requests each. SGLang replayed [the saved input file](baseline_metrics/aiperf/inputs.json); the two vLLM runs generated identical payload sequences (their session UUIDs differ).

| Runtime | Speculation | E2E p50 (ms) | TTFT p50 (ms) | ITL p50 (ms) | Inter-chunk p50 (ms) | Output tok/s | Acceptance length |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM `b6e7c1f1f` | off | 868.0 | 28.2 | 3.29 | 3.29 | 294.3 | — |
| vLLM `b6e7c1f1f` | DFlash, 15 draft tokens | 425.7 | 42.1 | 1.50 | 8.80 | 569.9 | 5.68 |
| SGLang 0.5.17 | off | 858.5 | 29.8 | 3.25 | 3.26 | 297.9 | — |
| SGLang 0.5.17 | DFlash, 16-token verify block | 294.5 | 34.1 | 1.02 | 5.99 | 829.2 | 5.70 |

The table uses the profiling phase only. vLLM DFlash acceptance length is `1 + 42762 accepted draft tokens / 9131 drafts = 5.683`. SGLang's profiling-phase `sglang:spec_accept_length` gauge averaged 5.700 across scrapes; its final value was 5.5, and its verify-call counter increased from 4 to 10281 over warmup plus profiling. Including warmups, the vLLM counter calculation gives 5.605, close to the issue's 5.602. All four measured runs finished 200 requests without errors; server counters show 220 streamed requests and 56320 generated tokens each, including warmup.

The two baselines differ by about 1% in median E2E latency. SGLang DFlash is about 1.45 times faster than vLLM DFlash in median E2E latency and output throughput, while their acceptance lengths are close. The largest visible differences are DFlash inter-chunk latency (8.80 versus 5.99 ms), TTFT (42.1 versus 34.1 ms), and ITL (1.50 versus 1.02 ms). This points toward runtime work per speculative step as a useful next profiling target; the measurements alone do not isolate a kernel or scheduler cause.

## Historical vLLM check

The issue's exact vLLM commit, `5b3762a7f`, was run with its matching precompiled wheel, PyTorch 2.11.0+cu130, and FlashInfer 0.6.14. On a separate 100-request GSM8K comparison with 20 warmups, baseline E2E p50 was 871.1 ms, DFlash E2E p50 was 496.5 ms, and DFlash acceptance length was 5.567 during profiling (5.502 including warmups). That pair is comparable within itself, but its request count differs from the primary 200-request comparison. The issue's reported 1.2 times speedup was not observed on this machine; the original commit produced about 1.75 times speedup, and the newer commit produced about 2.04 times. The issue used 1000 measured requests and 100 warmups, so these shorter runs do not establish its longer-run behavior.

## Artifacts and metric calculation

- `baseline_metrics/`, `dflash/`, `sglang_baseline/`, and `sglang_dflash/` contain raw `/metrics` snapshots and aiperf exports. The SGLang folders also contain `/get_server_info` snapshots. `original_baseline_valid/` and `original_dflash/` contain the historical comparison. `original_baseline/` records an initial failed run before FlashInfer was aligned to the historical pin.
- For vLLM, [the metrics collector](../benchmarks/issue_49730_metrics.py) subtracts the before snapshot from the after snapshot. Mean acceptance length includes the bonus token: `1 + delta(vllm:spec_decode_num_accepted_tokens_total) / delta(vllm:spec_decode_num_drafts_total)`. Draft acceptance rate and per-position acceptance rates are in each `acceptance_summary.json`. The profiling-only counter totals are in each aiperf `server_metrics_export.json`.
- The issue's `--disable-log-stats` flag was removed from both vLLM configurations because it suppresses engine metrics in this checkout. SGLang used `--enable-metrics`. SGLang's `--mem-fraction-static 0.8` and vLLM's `--gpu-memory-utilization 0.92` are different memory controls; memory pressure was negligible at concurrency 1.

The main comparisons use [vLLM baseline](baseline_metrics/aiperf/profile_export_aiperf.json), [vLLM DFlash](dflash/aiperf/profile_export_aiperf.json), [SGLang baseline](sglang_baseline/aiperf/profile_export_aiperf.json), and [SGLang DFlash](sglang_dflash/aiperf/profile_export_aiperf.json). The vLLM acceptance counter deltas are in [dflash/acceptance_summary.json](dflash/acceptance_summary.json), and the SGLang acceptance gauge is in [sglang_dflash/metrics_after.prom](sglang_dflash/metrics_after.prom).

## Replay command

With a server listening on port 8000 and serving the model as `qwen`, run the same 200 measured requests with:

```bash
/home/USER/.venv/bin/aiperf profile --model qwen --tokenizer Qwen/Qwen3.5-4B --url http://127.0.0.1:8000 --request-count 200 --warmup-request-count 20 --input-file issue_49730/baseline_metrics/aiperf/inputs.json --custom-dataset-type inputs-json --concurrency 1 --endpoint-type chat --streaming --output-artifact-dir issue_49730/replay
```

The vLLM server follows the issue command with `--disable-log-stats` removed. SGLang 0.5.17 baseline used `--model-path Qwen/Qwen3.5-4B --served-model-name qwen --tp-size 1 --dtype bfloat16 --quantization fp8 --context-length 32768 --mem-fraction-static 0.8 --enable-metrics`; its DFlash run added `--speculative-algorithm DFLASH --speculative-draft-model-path z-lab/Qwen3.5-4B-DFlash --speculative-num-draft-tokens 16`. All GPU runs were wrapped in `canhazgpu run --gpus 1 -- ...`.
