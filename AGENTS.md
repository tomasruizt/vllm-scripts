# Reuse existing infrastructure

Before implementing an experiment runner, collector, analysis or report, read [benchmarking/README.md](benchmarking/README.md) and search the shared packages and existing experiments for the operation you need.

- `benchmarking/aiperf.py`: AIPerf artifact ingestion, steady-window metrics, server counters and configurable validity gates.
- `benchmarking/artifacts.py`: JSONL reading and CSV/Parquet exports. Parquet uses the DuckDB CLI.
- `reporting/aiperf_report.py`: configurable AV off/on SPEED-Bench report builder. Qwen and XPress contain examples of thin configuration entry points.
- `reporting/plots.py`: series, individual observations, aggregation and PNG/SVG export.
- `reporting/model_tabs.py` and `reporting/b200-theme.css`: existing report composition and styling.

Import reusable code from these packages. Do not copy a report builder or import generic helpers from another experiment's `build_report.py` or `fit_cost_curves.py`. Existing compatibility exports support older consumers, they are not the interface for new code.

If a shared helper almost fits, extend it with explicit parameters and migrate the caller. Keep model settings, workload policy, architecture-specific checks and report wording in the experiment. Check existing runners before creating lifecycle, remote-job, workload-planning or profiler helpers, these are not yet covered by the shared packages.

Preserve metric definitions when migrating code. Distinguish whole-cell and steady-window metrics, mean and median aggregation, and paired gains and ratios of aggregates. Keep validity thresholds explicit. Verify numerical equivalence on saved artifacts when available, without starting a GPU job.

Run `~/.venv/bin/python -m unittest discover -s benchmarking/tests` after changing metric or export behavior. Read [reporting/AGENTS.md](reporting/AGENTS.md) before changing a report. Deploy the shared packages with scripts that import them, rather than copying a standalone experiment entry point.

Keep generated files in each experiment's ignored `results/`, `plots/` or `report/` directories. Track work and durable knowledge in the private Beads database, inherited repository instructions still apply.
