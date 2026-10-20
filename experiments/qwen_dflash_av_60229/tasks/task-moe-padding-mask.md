# Task: does AV's startup profiling skip MoE expert compute?

Read `README.md` in this folder first.

## Context

AV profiles step costs at startup through `_dummy_run` without `is_profile=True`. In `vllm/v1/worker/gpu/model_runner.py` dummy batches are built with `is_padding=not is_profile`, so every profiling token is marked as padding. `VLLM_MOE_SKIP_PADDING` defaults to 1 (`vllm/envs.py`), and the top-k routers (`vllm/model_executor/layers/fused_moe/router/fused_topk_router.py` and siblings, `_get_padding_mask`) then route padding tokens to no expert. If that happens, the profiled step cost excludes most MoE work and AV's cost table is too cheap.

Our runs used the `TrtLlmBf16ExpertsMonolithic` MoE backend (see the server log line "Using TrtLlmBf16ExpertsMonolithic MoE backend"), which routes inside its fused kernel and probably ignores the mask. Other backends (Triton, CUTLASS, modular FlashInfer) may honor it. This is unverified in both directions.

## What to do

1. Confirm from code which MoE backends consult `_get_padding_mask` / `is_padding`, and which do not (monolithic TRT-LLM).
2. Measure on the B300 with AV on, using the trace patch to log the startup cost curve (`../av-diagnostics.patch` + `../av_trace.py`, set `AV_TRACE_PATH`; see FINDINGS.md for how they are applied to a copy of the checkout):
   - default backend, `VLLM_MOE_SKIP_PADDING=1` vs `0`
   - a non-monolithic MoE backend (force it via the MoE backend flag or env var, find the right one in `vllm/model_executor/layers/fused_moe/`), `VLLM_MOE_SKIP_PADDING=1` vs `0`
3. Compare the profiled verify curves. If they differ, also run a short c64 benchmark with AV on for the affected backend in both settings to see whether AV's decisions and throughput change.

## Deliverable

- Which backends are affected, with file and line references.
- The cost curves (a few sizes are enough) for each setting, and any throughput difference.
- A proposed minimal fix (e.g. pass `is_profile=True` or force `is_padding=False` for AV profiling runs), with the risk it carries.
