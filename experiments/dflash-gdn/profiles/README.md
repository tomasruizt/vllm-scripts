# Short DFlash PyTorch profiles

Both engines served `Qwen/Qwen3.5-4B` with `z-lab/Qwen3.5-4B-DFlash` on one reserved H100. The client sent the same first GSM8K prompt from `../baseline_metrics/aiperf/inputs.json`, `temperature=0`, and separate `max_tokens=10` and `max_tokens=20` requests. Each server received a warmup request before profiling. vLLM used FP8, 15 speculative tokens, and `--max-num-seqs 8`; SGLang used FP8 and 16 draft tokens. Both used a 4096-token context for these short profiles. These startup settings differ from the earlier throughput benchmark; use that benchmark for unprofiled ITL.

| Engine | Output tokens | Profiled batch/execute calls | DFlash proposal/generation calls | Client time | Profiled batch/execute CPU span |
| --- | ---: | ---: | ---: | ---: | ---: |
| vLLM | 10 | 5 | 4 | 159 ms | 117 ms |
| vLLM | 20 | 6 | 5 | 172 ms | 133 ms |
| SGLang | 10 | 4 | 4 | 549 ms | 534 ms |
| SGLang | 20 | 5 | 5 | 86 ms | 72 ms |

The SGLang 10-token trace includes a 510 ms first profiled batch, apparently one-time profiler or kernel setup; it should not be treated as normal request latency. In the 20-token traces, vLLM's first `execute_model` spans 63 ms and its next four active calls span 17.6, 17.2, 16.8, and 17.4 ms. SGLang's first `run_batch` spans 42 ms and its next four calls span 7.9, 7.3, 7.4, and 7.4 ms. This supports the earlier observation that vLLM spends more time per DFlash loop in this configuration. These are CPU event spans from PyTorch profiler and can include asynchronous launch and profiler overhead; they are not isolated GPU kernel times.

## Where the profiled vLLM steps spend time

For each of vLLM's four steady `execute_model` calls in the 20-token trace, `MambaHybridModelState.prepare_attn` takes 14.8–15.2 ms. Its nested `build_attn_metadata` takes 14.6–15.0 ms and invokes `GDNAttentionMetadataBuilder.build` 24 times. Those 24 calls total 13.5–13.8 ms per step, about 0.51–0.78 ms each under the profiler. This is the dominant CPU span in vLLM's 16.8–17.6 ms execute calls; the corresponding SGLang steady `run_batch` calls span 7.3–7.9 ms, with about 0.8–0.9 ms across four `init_forward_metadata_out_graph` calls. Both engines used PyTorch's profiler with CPU/GPU activities and Python stacks. The traces point to repeated GDN attention metadata construction in vLLM as the largest measured difference. The unprofiled inter-chunk gap is only 2.8 ms larger for vLLM (8.80 versus 5.99 ms), so the 10 ms difference between profiled CPU spans must not be interpreted as the production latency difference. Profiling instrumentation, asynchronous GPU work, and the engines' different step boundaries limit that comparison.

The vLLM captures are in `vllm/tokens_10/` and `vllm/tokens_20/`. Each directory contains the engine's `rank0.*.pt.trace.json.gz`, the API server's `async_llm.*.pt.trace.json.gz`, and the request's `tokens_*.json` response with client timing. The SGLang engine traces are in `sglang/tokens_10/` and `sglang/tokens_20/`, with responses in `sglang/tokens_*.json`. Open the compressed trace files with Perfetto or Chrome tracing. Server startup and profiler messages are in `vllm/server.log` and `sglang/server_retry.log`.
