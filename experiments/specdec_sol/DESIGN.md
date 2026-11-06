# specdec_sol design notes: NCU data

How per-kernel NCU counters get into the SOL report, why the first design was wrong, and what the current one trades off. See [README.md](README.md) for usage.

## Goal

Every GPU kernel in the nsys timeline of a steady decode step should carry measured NCU counters (duration, SM and memory throughput, DRAM bytes), so that `x SOL = trace duration / (ncu duration x busiest-unit utilization)` is a measurement, not a model.
By default no analytic or semantic estimate is used (`sol.py --estimates` turns them on).

## Pipeline

1. **nsys pass.** A vLLM server under `nsys` with AIPerf load gives the unperturbed timeline.
2. **ncu pass.** A second vLLM server under `ncu` (`--kernel-id ::regex:.*:1`, `--filter-mode per-launch-config`) profiles the first launch of every kernel signature under AIPerf load. A signature is (mangled name, grid, block).
3. **ncu_fill pass.** A third server under `ncu` replays the nsys step shapes through the runner's dummy forward, for kernels the live pass never launched.
4. **ncu_step pass.** A fourth server under `ncu` profiles every launch of one step per shape. `sol.py` matches the k-th launch of a signature in the trace to the k-th in the profiled step.

Passes 3 and 4 are run on demand (`make fill-ncu`, `make step-ncu`), after `make analyze` has produced `steps.parquet`.

## Problems

### P1. The ncu server runs different shapes than the nsys server

Adaptive verification (AV) prices its choices from step timings taken at startup. Under ncu those timings are distorted, so the ncu server picked different verification lengths.
In the PR #65874 c1 run the nsys steady steps executed 4 tokens in 70% of steps and 9 in the rest, and the ncu pass only saw 9.
The MoE variant follows the token count (`routingIndicesBlockKernel` and `t128x8` expert GEMMs for 4 tokens, the `DynBlock` and `t128x16` variants for 9), so the 4-token kernels had no NCU row. 17% of c1 kernel time was affected.
Any runtime decision (AV, autotuning, chunked prefill, batch composition) can cause this. Re-running and hoping the shapes repeat is not reliable.
The autotuner was ruled out: both servers loaded the same FlashInfer autotune cache.

### P2. A signature does not include the problem size

NCU keeps only the first launch of each signature. The kernel name encodes the tile configuration, not M, N or K, and many kernels use a fixed persistent grid.
So one signature can serve layers that differ by orders of magnitude in bytes moved. In the PR c1 trace 14 of 39 GEMM signatures covered more than one work size, for example `nvjet_..._256x8_..._TNT` from 0.01 MB to 131 MB.
One launch's counters then describe all of them, which is wrong in both directions.

### P3. The fill replay itself can produce wrong counters

The first replay used zero input ids. All tokens routed to the same few experts, so MoE kernels read a fraction of the real bytes.
The c32 bmm with grid (8, 352, 1) took 144 µs in the trace but measured 32.7 µs and 34.5 MB in the replay. Applied as an exact match this gave x SOL 7.6 and a red box.
The live pass, with real routing, measured 131 µs and 895 MB for a nearby grid.

## Solution

- **Replay the nsys shapes.** `fill_shapes.py` derives one entry per (requests, padded tokens) of the steady steps. `SolProbe.sol_replay_shapes` runs the runner's dummy forward for each, then the LM head and sampler (`_dummy_sampler_run`), because `_dummy_run` stops before them.
- **Route like real tokens.** `randomize_inputs=True` spreads tokens over experts. `is_profile=True` marks tokens as real instead of padding. `valid_dummy_state_slots=True` keeps the GDN layers on their real kernels.
- **Profile every launch of one step per shape.** `profile=True` runs each shape unprofiled first (autotuning, compilation), then once inside a single `cudaProfilerStart/Stop` window. `spin_kernel` launches mark where each shape begins. The ncu flags drop `--kernel-id` and `--filter-mode`.
- **Match by occurrence.** `sol.py` gives the k-th launch of a signature in a trace step the counters of the k-th launch of that signature in the profiled step of the same shape (`ncu_match = step launch`). Other kernels fall back to the per-signature match, then to the same kernel and block at another grid.
- **Make the gap visible.** The report check `kernel time without an NCU row` states the unmatched share.

Result on the PR run: c1 has 0.4% of steady kernel time without an NCU row. At c32 the large bmm ops moved from x SOL 7.6 to 1.3 (77 to 82% of peak).

## Tradeoffs and limitations

- **Cost.** Profiling every launch runs at about 56 launches per minute, so about 1.5 hours for c1. It scales with shapes and kernels per step. The signature pass takes about 40 minutes for both levels.
- **Not the live workload.** Counters come from a dummy forward. Inputs, routing, contexts and KV contents are synthetic. Random routing is closer to reality than zeros, but real routing is skewed.
- **Eager replay, graph trace.** The replay runs eagerly (`is_profile`) while the trace is a CUDA graph. The kernels should be the same, but this is not proven.
- **Occurrence matching can be wrong silently.** It assumes launch order is the same in the replay and the trace. Across parallel streams, start order can swap neighbouring layers. The check so far is the trace-to-NCU duration ratio (median 0.59), which catches large mismatches only. If the launch counts differ, the affected kernels fall back to the signature match.
- **Coverage is limited to replayed shapes.** At c32 only the main decode shape was profiled. Prefill-containing steps still have 5.5% of kernel time without a row and 18% matched by another grid.
- **ncu isolation.** ncu replays each kernel alone, without clock control, cache control or concurrency. Trace durations are about 0.6 of the ncu durations at the median, so x SOL is relative to the ncu run, not an absolute bound.
- **Coupling to vLLM internals.** The replay uses `_dummy_run`, `_dummy_sampler_run` and `max_num_reqs`. A refactor can break it. The sentinel logic assumes `spin_kernel` appears in the ncu CSV.
- **Single run.** Run-to-run variation of the per-launch counters has not been measured.

## Open items

- Profile the c32 prefill shapes, or exclude prefill-containing steps from the steady set.
- Validate the occurrence matching on more cases, for example by comparing the sequence of kernel names between replay and trace per step.
- Repeat a profile to measure variation.
- Run the step pass for the base arm of the PR #65874 comparison.
