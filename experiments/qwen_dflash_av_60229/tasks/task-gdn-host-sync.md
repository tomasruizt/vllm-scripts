# Task: does the GDN prefill path force a host sync per layer?

Read `README.md` in this folder first.

## Context

Steps with a prefill run eagerly here and are CPU-bound: ~90 ms of wall time for ~19 ms of GPU work, ~2.4 ms of CPU per layer. A py-spy profile attributes ~38% of that CPU time to the GDN linear-attention core (30 of the 40 layers), which runs eagerly even with piecewise CUDA graphs. Raw data and the analysis script: `../results/prefill-step-profile/`, `../analyze_pyspy.py`, `../profile-prefill-step.sh` (`make profile-prefill`).

One unverified suspect: in `vllm/model_executor/layers/gdn/qwen_gdn_linear_attn.py` around line 1510, `initial_state[~prefill_has_initial_state, ...] = 0`. Boolean-mask indexing with a GPU mask needs the number of selected elements on the host, which forces a device-to-host sync. Once per GDN layer that stops the CPU from running ahead of the GPU, which hurts exactly in the CPU-bound eager regime.

## What to do

1. Confirm whether the mask lives on the GPU in this path, and whether the line triggers a sync (torch profiler or nsys: look for `cudaStreamSynchronize` / `cudaMemcpy` DtoH / `aten::nonzero` per GDN layer). `torch.cuda.set_sync_debug_mode("warn")` around one prefill step is a quick check.
2. Search the rest of the GDN prefill path for other syncs (`.item()`, `.tolist()`, `nonzero`, boolean indexing, `.cpu()`).
3. Prototype a sync-free version (e.g. `torch.where` or multiplying by a float mask) in a copy of the checkout, check that outputs match (e.g. greedy generations on a few prompts are identical), and measure:
   - CPU and wall time of a solo 1,080-token prefill step (`make profile-prefill`, torch profiler mode)
   - a short c128 run with default capture sizes, AV off (throughput and the prefill-step band from the ITL histogram)

## Deliverable

- List of syncs found in the GDN prefill path (file, line, cost per step).
- Prefill step time before and after the prototype, and the c128 throughput change.
- The diff of the prototype and a note on correctness checks.
