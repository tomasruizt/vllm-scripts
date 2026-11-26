# NVIDIA Nsight Perf SDK for AI: trial on vLLM

The [Nsight Perf SDK for AI](https://github.com/NVIDIA/nsight-perf-sdk-for-ai) (beta preview update 1, 2026-11-17) is an always-on GPU and CUDA tracer. A daemon runs next to the workload, so a serving process can be traced without restarting it under a profiler, and the trace exports to Perfetto or to DuckDB and parquet. This trial traces a vLLM server under an AgentX workload on one B200.

Status: the setup works and records GPU kernel timings. A complete traced serving run with the profiling permission is still pending (see "Runs").

## What it takes on a B200 node

Node: kernel driver 595.71.05. Image: `vllm/vllm-openai:nightly` with CUDA 13.0. The SDK asks for driver 620 and CUDA 13.5.

- **Prebuilt release:** the x86_64 tarball works as is, with daemon, server and exporter in one container (user-local daemon, no installation).
- **CUDA 13.5 user-mode driver:** nothing is traced until the server loads it. The compat package (`cuda-compat-13-5`) on `LD_LIBRARY_PATH` of the server is enough, the kernel driver can stay at 595.
- **`CAP_PERFMON` for the daemon:** needed for GPU kernel timings and PM counters. Without it, the SDK's own smoke test reports 0 GPU kernel events, and the trace only has the CPU side (API calls, launches with kernel names) and memcpy timings. With it, the smoke test passes on this driver. The alternative is to open profiling to all users on the node (`NVreg_RestrictProfilingToAdminUsers=0`).
- **OpenShift:** the capability needs a service account with a matching SCC. The one we used forbids projected volumes, so the pod sets `automountServiceAccountToken: false`, otherwise admission fails.
- **NVTX:** vLLM emits no NVTX ranges, so the trace cannot attribute GPU time to step types (prefill, decode) yet. Push-pop ranges around the step would enable that.

## Runs

Model: `inference-optimization/GLM-5.3-Flash-2.6B-A2.3B-small` (a small clone of GLM-5.3-Flash without MTP weights, so no speculative decoding), one B200, AgentX at c8 for 300 s.

| Run | Daemon capability | Outcome |
|---|---|---|
| `nsight-c8-1` | none | Complete: 54 requests, 0 errors, 14 snapshots, 191 MB of parquet. CPU-side trace only. |
| `nsight-c8-2`, `nsight-c8-4` | `CAP_PERFMON` | Trace with GPU timings, but it only covers server startup: the AgentX client failed on its setup (incomplete HF cache on a fresh disk, fixed in the scripts since). |

What `nsight-c8-1` shows about vLLM (324 s of serving, CPU side):

- 88,148 CUDA graph launches from 104 distinct graphs, so about 272 steps/s.
- About 34 immediate kernel launches per step outside the graph (sampler and bookkeeping, e.g. `TopPSamplingFromProbKernel`, `cunn_SoftMaxForward`, `_token_request_mapping_kernel`), plus about 9 async memcpys.
- The driver calls take about 190 µs of CPU time per step (5% of the step), tracing overhead included. There is no untraced baseline for this run.

What the `CAP_PERFMON` traces add: GPU start and end times for every kernel (97,298 CUDA graph node launches and 12,044 immediate launches, all populated) and 419,768 PM counter samples.

## Running it

The model disk, the CPU pod and the AgentX client come from [../glm53_agentx](../glm53_agentx). Both folders take the same `make` variables for cluster, namespace, node and resource prefix. The namespace must have a service account (`PROFILER_SA`) that may add `CAP_PERFMON`.

```bash
make -C ../glm53_agentx box prepare NS=<namespace>     # once per namespace: disk, weights, client, traces
make prepare NS=<namespace>                            # SDK and CUDA 13.5 compat package onto the disk
make run CONC=8 DURATION=300 RUN_ID=nsight-c8-1 NS=<namespace>
make fetch-results RUN_ID=nsight-c8-1 NS=<namespace>   # parquet export and logs into results/ (git-ignored)
```

The trace tables are in `results/<run>/nvtrace/*/parquet_export/`, with `schema.sql` next to them. Example, CPU time per driver call:

```bash
duckdb -c "select driver_api_function_name, count(*) n, round(sum(end_utc_ns - start_utc_ns) / 1e9, 2) cpu_s
           from 'cuda_driver_api_calls.parquet' group by 1 order by cpu_s desc limit 10"
```
