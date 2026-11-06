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
passes: [nsys, ncu]  # ncu supplies counter references when semantic metadata is missing
```

2. Launch, wait, and build the report:

```bash
make launch CONFIG=configs/qwen36-35b-a3b-dspark-av.yaml RUN_ID=qwen36-dspark-av-20261103
make status RUN_ID=qwen36-dspark-av-20261103
make watch RUN_ID=qwen36-dspark-av-20261103    # blocks until every level finished
make report RUN_ID=qwen36-dspark-av-20261103   # report/<RUN_ID>/index.html, open it in a browser (it needs JavaScript)
```

Each concurrency level runs in its own `chg` reservation, in parallel. A level takes ~10 min for the nsys pass and ~40 min for the ncu pass.
Semantic SOL needs no ncu. `make report NCU_FROM=results/<other-run>` reuses counters from an earlier run of the same workload (kernels are matched by mangled name, grid and block). For a broader profile bank, call `sol.py` with repeated `--ncu <CSV>` arguments from the same GPU type, with the preferred workload first. GPU names are checked when recorded in the CSV. Selected CSVs are saved in `meta.json` for subsequent local rebuilds and each op retains its counter source. Identical launch signatures can still process different data, so cross-workload reuse remains a reference.

## Method

Per concurrency level, two server runs under AIPerf load:

1. **nsys**: unperturbed timeline with `--cuda-graph-trace node`. [sol_probe.py](sol_probe.py) wraps the runner's `execute_model()`, `sample_tokens()`, input preparation, sampling, the speculator's `propose()` and `CUDAGraph.replay()` in NVTX ranges, found on the live objects, so it works with Model Runner V1 and V2 and any draft method. It also marks each step with its batch shape. An AIPerf run before the capture gives the reference TPOT and acceptance length.
2. **ncu** (optional, own reservation, runs in parallel): SOL throughput and DRAM bytes for the first launch of every kernel signature (`--filter-mode per-launch-config`) after `/start_profile`, with a small KV pool so kernel replay stays on the GPU. It stops after `ncu_idle_seconds` without a new signature. With TP>1 the workers replay in lockstep (`--communicator shmem --lockstep-kernel-launch`, ncu >= 2026.1), otherwise collectives deadlock. Not yet tested with TP>1.

The live ncu pass can miss kernels of the nsys run, because runtime decisions (adaptive verification prices its choices from timings taken at startup, which ncu distorts) change the executed shapes. The report check `kernel time without an NCU row` shows the share. `make fill-ncu RUN_ID=<RUN_ID>` replays the nsys run's step shapes (from `steps.parquet`, see [fill_shapes.py](fill_shapes.py)) through the runner's dummy forward under ncu into `c<N>/ncu_fill/`. Then `make report RUN_ID=<RUN_ID>` merges it, the live ncu profile has priority.

[sol.py](sol.py) charges every GPU op its busy time plus the idle gap before it, so op costs sum to the step time. Steady steps are pure decode steps with the most common request count. `tokens` is the scheduled verify batch (1 + the number of drafts), not the tokens a step generates, so the report states the acceptance length, the generated tokens per step and the per-token cost the traced step implies next to the reference TPOT. A steady step is flagged as **stalled** when the GPU idle its host caused exceeds max(1 ms, 5x the median steady host gap). The 1 ms floor decides in practice, and it is far above launch latency: the waterfall stays a mean over every steady step, and the report gives the median-step ratio and the mean without the stalled steps, because a few stalls can dominate the mean while the timeline shows the median step.

By default the SOL is NCU data only: each kernel's measured duration times its busiest unit's utilization, and kernels without an NCU row have no SOL. The analytic step SOL and the semantic kernel models below are opt-in, `make analyze ESTIMATES=1` (or `sol.py --estimates`).

The **analytic SOL** of a step needs no ncu. The probe reports weights, logical linear dimensions and precision, cache sizes and a measured BF16 GEMM ceiling. The optimistic forward reference reads weights, existing KV and initial recurrent state once. It excludes intermediate activations, checkpoint writes and attention/recurrence/nonlinear arithmetic. MoE traffic assumes uniform routing across requests and shared experts across a request's draft tokens. Draft weight reads count one forward per graph replay, while draft FLOPs remain unknown until a draft execution shape is recorded. Legacy Llama metadata retains a partial memory reference without inventing a compute ceiling.

The **per-kernel SOL** uses the shared [semantic registry](../../benchmarking/roofline.py), with memory and compute bounds priced separately by the hardware profile:

1. D2D copies count both a payload read and write, memset counts a write. Host transfers require a link profile and are otherwise unsupported.
2. [kernel_models.py](kernel_models.py) associates known kernels with semantic layers using execution order. This association remains inferred and is disclosed. Unknown kernels, reductions, activation GEMMs and sampling helpers receive no generic activation-sized fallback. Stateful references exclude intermediate checkpoints and declare missing arithmetic.
3. Kernels without semantic metadata use an NCU resource-demand reference: replay duration times the larger compute/memory utilization. Exact signatures are labeled `counter-reference`. Matching only the kernel and block transfers utilization to the current duration and is labeled `inferred-counter-reference`. These references describe measured implementation work and exclude latency, they are not implementation-independent bounds.

Kernels are gray only when neither semantic metadata nor matching counters are available. Groups containing unknown estimates show `n/a` for SOL and excess time, retaining measured durations, and their `SOL coverage` column says which share of the group's busy time is mapped, so `n/a` is not read as unmapped. Bounds are not clipped to measured time. Assumptions, source and quality are retained in Parquet and timeline tooltips.

New traces include `sol/execution` marks after input preparation, with logical token totals, padded sizes and available host query/context lengths. No GPU readback or synchronization is added. Multi-request AV query splits may be GPU-only, so the probe leaves those lengths unknown rather than reporting the CPU placeholder distribution as exact. Older traces retain scheduled shapes and explicitly mark missing execution shapes. The raw scheduler token field remains unchanged.

Set `nsys_shapes: true` for a shape diagnostic. This enables NSYS `--pytorch=autograd-shapes-nvtx` and starts collection in the worker's `capture_model()` before CUDA graph construction. The runner marks a later AIPerf measurement window, so startup and graph construction supply metadata but are excluded from steady timing summaries. The usual unprofiled reference run is omitted because collection is already active. This requires a vLLM version that calls the worker extension at startup, otherwise the run fails explicitly. Annotation overhead makes this a diagnostic, retain the normal run for performance comparisons.

With `nsys_shared_server: true`, all requested concurrency workloads use one server configured for the largest concurrency. Each workload retains separate AIPerf artifacts and tagged measurement windows. Smaller levels link to the shared trace, so graph construction is captured once. Analysis selects the requested window before finding steady steps. This avoids repeated model loading, while the recorded serve command makes the shared maximum sequence count explicit. NCU jobs remain independent.

[benchmarking/nsys_shapes.py](../../benchmarking/nsys_shapes.py) parses native operator input sizes. Graph kernels follow process-scoped `originalGraphNodeId` clone links, eager kernels use their enclosing launch annotation. Names and occurrence order are never used to transfer shapes. Ambiguous or missing capture provenance remains unmapped. Operators spanning several GPU kernels retain shape metadata but receive no duplicated per-kernel formula. Supported single-kernel matrix operators can use matching layer precision declarations. Native annotations omit dtype, so arbitrary operators are not assigned the model's activation dtype. For a controlled workload with a known uniform dtype, `sol.py --shape-dtype bf16` declares that assumption explicitly and retains it on rebuilds. Pointwise and reduction estimates remain partial when arithmetic or output metadata is missing. Overview checks give busy-time coverage for input shapes and shape-based estimates separately.

CPU validation: `PYTHONPATH=../.. ~/.venv/bin/python -m unittest discover -s . -p test_estimates.py`. Shared estimator tests run with `~/.venv/bin/python -m unittest discover -s benchmarking/tests` from the repository root.

[report.py](report.py) builds the HTML. The overview lists checks (steady steps, GPU stalls, SOL coverage, phase coverage, inferred launches), so a run that went wrong shows in the report.

## Limitations

- Under TP the analysis follows one rank. Semantic communication estimates require an explicit link bandwidth.
- Sparse or compressed attention (e.g. DeepSeek-V4) does not fit the per-layer KV model and stays unmapped.
- Graph kernels without a CUPTI launch record get the phase of the latest graph launch without kernels of its own (inferred, marked in tooltips).
- The ncu SOL is measured under kernel replay with warm L2 (`--cache-control none`) and a smaller KV pool.

## Analyzing results

Per run and concurrency level, everything is in `report/<RUN_ID>/c<N>/` (local, git-ignored, built by `make report`). Raw inputs (nsys sqlite, ncu CSV, AIPerf, server logs, `nsys/model-info.json`) are in `results/<RUN_ID>/c<N>/`.

- `steps.parquet`: one row per engine step. `steady` marks the steps the report averages over. `reqs`, `tokens` (the scheduled verify batch, 1 + drafts) and `context` (summed context tokens) are the batch shape. `cost_ns` = busy + idle GPU time charged to the step. `target_sol_ns`, `draft_sol_ns`, `analytic_sol_ns` are the step-level analytic SOL, `draft_forwards` the draft forwards counted per step.
- `ops.parquet`: one row per GPU op (kernel, memcpy, memset), joined to `steps` on `step`. `phase` is the nested CPU call that launched it (e.g. `GPUModelRunner.execute_model>CUDAGraph.replay`), `name` the kernel's short name, `signature` its mangled name, grid and block. Time: `dur_ns` (individual duration), `busy_ns` (overlap counted once), `gap_ns` (idle before the op, kind in `gap_kind`, host-bound part in `host_gap_ns`). SOL: `sol_ns`, `sol_source`, `sol_quality`, `sol_assumptions`, separate memory/compute references, `model_family` and `model_sol_ns` (semantic), `sol_pct`, `kernel_sol_ns`, `ncu_match` (ncu). Timeline colors use `dur_ns / sol_ns`, while attributed cost remains `busy_ns + gap_ns`.
- `meta.json`: GPU peak bandwidth, the shown steps, and `checks` (steady rule, SOL coverage, GEMM mapping per graph variant). Read the checks first, they say what the numbers rest on.
- `operators.parquet`: native PyTorch operator annotations with input sizes and CPU range identities. `ops.parquet` retains each associated operator, input sizes, provenance and kernel count, plus whether that metadata supplied its estimate.
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

## Greedy verifier microbenchmark

[bench_greedy_rejection.py](bench_greedy_rejection.py) compares pinned main and PR sampling modules using production UVA metadata, without loading a model. It times the stats, rejection, resample and insert kernel chain together with FlashInfer's CUPTI helper, CUDA graph replay and cold L2. Defaults reproduce the B300/H100 comparison: BF16, vocabulary 248320, 1/32/128 requests, 3/7 drafts plus bonus, three alternating timing calls per case. The masked scenario retains top-20 logits before timing and enables masked-block skipping in both arms. Token IDs and counts must match before timing.

Use a CUDA vLLM environment with the matching Python support modules (including `draft_logits.py`), compiled extensions, FlashInfer and `cupti-python>=13`. Timing fallback raises an error. Both revisions must be available in `VLLM_SOURCE`. Run from this directory on each GPU architecture, using a fresh run ID:

```bash
VLLM_SOURCE=/path/to/vllm
MAIN_COMMIT=ca8bd4c0550655b3c15e92b94782c5909da65e6c
PR_COMMIT=13d298f3555a38d5239bf4e11e9e07953be62aa6
RUN_ID=greedy-b300-20261104
mkdir results/$RUN_ID
git -C "$VLLM_SOURCE" show "$MAIN_COMMIT":vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py > results/$RUN_ID/main_utils.py
git -C "$VLLM_SOURCE" show "$PR_COMMIT":vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py > results/$RUN_ID/pr_utils.py
setsid nohup chg run --gpus 1 --timeout 10m -- ~/.venv/bin/python bench_greedy_rejection.py \
    --main-module results/$RUN_ID/main_utils.py --pr-module results/$RUN_ID/pr_utils.py \
    --main-commit "$MAIN_COMMIT" --pr-commit "$PR_COMMIT" --output results/$RUN_ID \
    > results/$RUN_ID/job.log 2>&1 < /dev/null &
```

`metadata.json` records versions and source hashes. `measurements.jsonl` contains one row per timing call. `DONE` marks successful completion. Fetch remote outputs with `make fetch-greedy-results REMOTE=<host> RUN_ID=<run>`, then combine runs for the same pinned comparison:

```bash
make analyze-greedy GREEDY_INPUT='results/greedy-*/measurements.jsonl' GREEDY_OUTPUT=results/greedy-combined
```

[analyze_greedy_rejection.sql](analyze_greedy_rejection.sql) writes a combined long-format Parquet file and per-case, summary and repeat-range CSVs. Table ranges span accepted-prefix case medians. `avg_delta_us` is the arithmetic mean of `PR - main` case medians, so improvements are negative.
