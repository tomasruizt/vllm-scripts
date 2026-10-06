Build the throughput–interactivity report with `~/.venv/bin/python experiments/nemotron_av/analysis/build_report.py --source /path/to/sweep/results.json` from the `vllm-scripts` root.
The source directory must include each run’s before/after Prometheus snapshots.
Subsequent rebuilds need no arguments: the report archives its inputs locally.
To include another draft length, add `--comparison /path/to/other/sweep/results.json`.
Separate K=7 and K=15 into tabs, each containing its own throughput–interactivity plot and results table; select K=7 initially.
Both plots use the same axis limits and retain amber/green for AV off/on.

The DSpark sweeps use the BF16 Nemotron 3.5 Lightning target and NVIDIA’s separate NVFP4-DSpark checkpoint.
K=7 uses seven draft tokens plus the anchor, matching the checkpoint’s advertised eight-slot block; K=15 evaluates a longer block with the same checkpoint.
The drafter uses W4A16_NVFP4 with selected modules unquantized; it has no confidence head, so AV uses the online acceptance estimator.
Target, drafter, and draft length are rendered from saved server commands.

Interactivity is the reciprocal of mean request TPOT, pooling Prometheus sum/count deltas across three runs; throughput is the arithmetic mean of per-run throughput.
Show standard deviations only as plot error bars, never in tables.
Follow `../../../reporting/AGENTS.md` for styling and browser verification.
