# Task: measured AV cost curve on DeepSeek-V4.1-Flash-1017 (TP4 + EP)

Read `README.md` in this folder first, then the 2026-10-20 sections of `../FINDINGS.md` (rh-dte, rh-3js) and the report `../report/measured-cost-curve/index.html` (local, git-ignored).

## Question

On Qwen3.5-35B-A3B + DFlash, replacing AV's startup cost table with a curve measured during serving gave +14% at c64 and +11% at c128, −6% at c1, flat elsewhere (provisional, not interleaved, noisy baseline). The native startup table was flat for small steps (rh-g6f). Does the same hold on a large, widely used model where AV is native: DeepSeek-V4.1-Flash-1017 with its built-in DSpark draft head?

## Setup

- Model: `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision `1bae0e10` (FP8, FP4 experts, ~763B params). Download to the shared cache under `/data/tomasruizt/hub_cache` (check it is writable first).
- vLLM: upstream `main`, pinned to one commit at the start and recorded. AV with DSpark is upstream, so no PR branch. Note `main` profiles AV at `VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=8192` by default. Keep the default and record it.
- Parallelism: **TP4 + EP** on 4× B300 (`--tensor-parallel-size 4 --enable-expert-parallel`), no data parallelism. This matches the [AV blog post](https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification) setup (TP8 + EP on V4-Pro) and keeps one AV decision per step.
- Speculative config: `{"method":"dspark","num_speculative_tokens":K,"enable_adaptive_verification":true}`. The checkpoint has `dspark_block_size: 5`, the blog used K=7 on V4-Pro. Check which K vLLM accepts for V4.1 and pick the largest valid one, record why.
- Extended CUDA graph capture sizes as in `../run-steady.sh` (`cg_sizes`), so eager prefill steps don't dominate. Confirm graph capture memory still leaves enough KV cache for c256.
- Workload: SPEED-Bench throughput-1K with AIPerf 0.13.0, same template as `../full-curve-sweep.yaml` (natural EOS, max 1,024 output tokens, prefix caching off). Real prompts only, never random tokens.
- GPUs: reserve via `chg run` / canhazgpu, launch asynchronously, never interrupt other users' jobs.

## Stage 1: cheap check (go / no-go)

1. Dump the native startup table: `../dump-cost-tables.sh` with `TP=4`, plus a way to pass `--enable-expert-parallel` and the DSpark config (extend the script, keep the diff small).
2. Port the hooks: re-anchor `../install_full_curve.py` to the pinned `main` (AV's `get_num_tokens` / `set_cost_curves` and the model runner), keep `../av_full_curve.py` unchanged where possible. Add the native table dump to the hook so every run records the table it used.
3. Short calibration with randomized budgets at c1, c32 and c128 (one repeat each), then plot native vs measured with `make plot-cost-curves` (generalize the target's paths).
4. Decision: if native and measured disagree materially in relative prices (the argmax only depends on ratios), go to stage 2. If they agree, stop and report.

## Stage 2: full experiment (only after a go)

- Calibrate c1/c8/c32/c128/c256 (two repeats) plus targeted large-token prefills up to the scheduler's max, freeze the curve with `freeze_curves` (checksum, no fallback, fail closed outside the measured domain).
- Sweep c1, c8, c32, c64, c128, c256 on one persistent server, switching native/measured through the control file. **Interleave the two policies run by run** (the Qwen sweep ran all measured, then all native, see rh-bpx). Six runs per cell, steady-window metrics with the existing validity gates.
- Estimate total sequential runtime before launching and size each measurement to ~30 s.

## Deliverable

- Table per c: throughput, TPOT, AL, median and min–max for native and measured.
- Throughput vs interactivity plot (reuse `../plot_av_full_sweep.py`) and the native vs measured cost-curve figure.
- Full `vllm serve` and AIPerf commands, pinned vLLM commit, model revision, K.
- A dated section in `../FINDINGS.md`. Fetch results with a `make fetch-*` target, analyze locally with DuckDB. Publish nothing.
