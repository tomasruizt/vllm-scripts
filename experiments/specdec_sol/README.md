# Spec-decode SOL report

How far is one steady-state spec-decode step (target verify + draft) from speed of light, and where is the gap?
For any vLLM spec-decode workload on one node, the pipeline produces an offline HTML report: a zoomable timeline of one step with every kernel colored by its distance from SOL, a waterfall of the step cost, and tables by CPU phase, kernel category and kernel.

## Run a workload

1. Write a workload config in `configs/` (git-ignored, it holds machine-specific paths). Only `name`, `model` and `serve_args` are required, everything else has defaults (see `DEFAULTS` in [run.py](run.py)):

```yaml
name: Qwen3.6-35B-A3B + DSpark K=8 + AV
model: Qwen/Qwen3.6-35B-A3B
serve_args:  # everything except --port, --host, --max-num-seqs (= concurrency) and the profiler flags
  - --served-model-name=qwen
  - --language-model-only
  - --max-model-len=8192
  - --no-enable-prefix-caching
  - --disable-uvicorn-access-log
  - --speculative-config={"method":"dspark","model":"RedHatAI/Qwen3.6-35B-A3B-speculator.dspark","num_speculative_tokens":8,"enable_adaptive_verification":true}
env: {HF_HUB_OFFLINE: "1", VLLM_USE_V2_MODEL_RUNNER: "1"}  # plus HF_HUB_CACHE, PYTHONPATH for another vLLM checkout
request_extra: {chat_template_kwargs: {enable_thinking: false}}
dataset: {path: <SPEED-Bench throughput_1k.jsonl>, format: speed_bench_throughput_1k}
gpus: 1  # tensor parallel size, with gpus > 1 the ncu pass needs ncu >= 2026.1 (config key ncu)
concurrencies: [1, 32]
passes: [nsys, ncu]  # ncu adds measured per-kernel SOL for kernels without an analytic model
```

2. Launch, wait, and build the report:

```bash
make launch CONFIG=configs/qwen36-35b-a3b-dspark-av.yaml RUN_ID=qwen36-dspark-av-20261103
make status RUN_ID=qwen36-dspark-av-20261103
make watch RUN_ID=qwen36-dspark-av-20261103    # blocks until every level finished
make report RUN_ID=qwen36-dspark-av-20261103   # report/<RUN_ID>/index.html, open it in a browser (it needs JavaScript)
```

Each concurrency level runs in its own `chg` reservation, in parallel. A level takes ~10 min for the nsys pass and ~40 min for the ncu pass.
With `passes: [nsys]` the report has no SOL. `make report NCU_FROM=results/<other-run>` reuses the ncu data of an earlier run of the same workload (kernels are matched by mangled name, grid and block).

## Method

Per concurrency level, two server runs under AIPerf load:

1. **nsys**: unperturbed timeline with `--cuda-graph-trace node`. [sol_probe.py](sol_probe.py) wraps the runner's `execute_model()`, `sample_tokens()`, input preparation, sampling, the speculator's `propose()` and `CUDAGraph.replay()` in NVTX ranges, found on the live objects, so it works with Model Runner V1 and V2 and any draft method. It also marks each step with its batch shape. An AIPerf run before the capture gives the reference TPOT and acceptance length.
2. **ncu** (optional, own reservation, runs in parallel): SOL throughput and DRAM bytes for the first launch of every kernel signature (`--filter-mode per-launch-config`) after `/start_profile`, with a small KV pool so kernel replay stays on the GPU. It stops after `ncu_idle_seconds` without a new signature. With TP>1 the workers replay in lockstep (`--communicator shmem --lockstep-kernel-launch`, ncu >= 2026.1), otherwise collectives deadlock. Not yet tested with TP>1.

[sol.py](sol.py) charges every GPU op its busy time plus the idle gap before it, so op costs sum to the step time. Steady steps are pure decode steps with the most common request count.

The **analytic SOL** of a step needs no ncu. After startup the probe reports, from the live model: dense and expert weight bytes and elements, KV bytes per token, recurrent (GDN, Mamba) state bytes per request, top-k, and the measured BF16 GEMM rate. Per forward the bound is `max((weights + KV + state) / peak DRAM BW, 2 x active params x tokens / GEMM rate)`. MoE experts count as far as the R requests touch them under uniform routing (`E * (1 - (1 - k/E)^R)`, the tokens of one request route alike), and the draft does one forward per CUDA graph replay inside `propose()`. Activations and attention FLOPs are left out, so the bound is optimistic.

The **per-kernel SOL** is the first estimate in `sol_sources()` that knows the kernel:

1. copies: bytes / peak DRAM BW.
2. kernels: the analytic models of [kernel_models.py](kernel_models.py). Kernels are classified by name into families (GEMM, MoE expert GEMM, attention, KV write, SSM state update, conv, communication, activation GEMM, elementwise) and mapped in program order onto the layer inventory: GEMMs onto linear layers (with fused or GEMM-less layer classes found by a search that must give every layer class a consistent kernel), expert GEMMs onto MoE layers, attention onto KV-cache layers. Each bound is `max(bytes / peak DRAM BW, FLOPs / GEMM rate)`. Families that do not line up get no model and show in the report's checks.
3. kernels: ncu duration x max(SM %, memory %), the time at the peak of the busiest unit, matched by signature (or by name and block size for data-dependent grids). It needs no model, but cannot tell redundant work from useful work.

Ops without any estimate are reported as "busy time without a SOL estimate".

[report.py](report.py) builds the HTML. The overview lists checks (steady steps, SOL coverage, phase coverage, inferred launches), so a run that went wrong shows in the report.

## Limitations

- Under TP the analysis follows one rank. Communication kernels use a fixed NVLink bandwidth (900 GB/s).
- Sparse or compressed attention (e.g. DeepSeek-V4) does not fit the per-layer KV model and stays unmapped.
- Graph kernels without a CUPTI launch record get the phase of the latest graph launch without kernels of its own (inferred, marked in tooltips).
- The ncu SOL is measured under kernel replay with warm L2 (`--cache-control none`) and a smaller KV pool.

## Analyzing results

Per run and concurrency level, everything is in `report/<RUN_ID>/c<N>/` (local, git-ignored, built by `make report`). Raw inputs (nsys sqlite, ncu CSV, AIPerf, server logs, `nsys/model-info.json`) are in `results/<RUN_ID>/c<N>/`.

- `steps.parquet`: one row per engine step. `steady` marks the steps the report averages over. `reqs`, `tokens`, `context` (summed context tokens) are the batch shape. `cost_ns` = busy + idle GPU time charged to the step. `target_sol_ns`, `draft_sol_ns`, `analytic_sol_ns` are the step-level analytic SOL, `draft_forwards` the draft forwards counted per step.
- `ops.parquet`: one row per GPU op (kernel, memcpy, memset), joined to `steps` on `step`. `phase` is the nested CPU call that launched it (e.g. `GPUModelRunner.execute_model>CUDAGraph.replay`), `name` the kernel's short name, `signature` its mangled name, grid and block. Time: `busy_ns` (overlap counted once), `gap_ns` (idle before the op, kind in `gap_kind`, host-bound part in `host_gap_ns`). SOL: `sol_ns` with its origin in `sol_source`, `model_family` and `model_sol_ns` (analytic), `sol_pct`, `kernel_sol_ns`, `ncu_match` (ncu). Cost of an op = `busy_ns + gap_ns`, lost = cost - `sol_ns`.
- `meta.json`: GPU peak bandwidth, the shown steps, and `checks` (steady rule, SOL coverage, GEMM mapping per graph variant). Read the checks first, they say what the numbers rest on.
- `summary.md`: the waterfall and tables as markdown. `index.html` is the interactive report, `steps.perfetto.json` the shown steps for ui.perfetto.dev.

Example, lost time per step by phase and kernel family over steady steps:

```sql
duckdb -c "
WITH s AS (SELECT step FROM 'report/<RUN_ID>/c1/steps.parquet' WHERE steady)
SELECT phase, model_family, count(*) / (SELECT count(*) FROM s) AS ops_per_step,
       sum(busy_ns + gap_ns - coalesce(sol_ns, 0)) / 1e3 / (SELECT count(*) FROM s) AS lost_us_per_step
FROM 'report/<RUN_ID>/c1/ops.parquet' WHERE step IN (SELECT step FROM s)
GROUP BY ALL ORDER BY lost_us_per_step DESC LIMIT 20"
```
