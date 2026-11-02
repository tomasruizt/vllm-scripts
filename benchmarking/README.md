# Shared benchmark infrastructure

Use this layer before adding experiment-specific copies. Run scripts from a checkout, or set `PYTHONPATH` to the `vllm-scripts` root. There is no installation step. Local analysis uses `~/.venv/bin/python`, NumPy and the DuckDB CLI. Reports also require Matplotlib.

| Operation | Reusable API |
|---|---|
| Read AIPerf summary, server metrics and request records | `benchmarking.aiperf.read_artifacts(path, phase_name=None)` |
| Select completed requests | `artifacts.completed` |
| Steady-window throughput and latency | `benchmarking.aiperf.steady_window(requests, concurrency)` |
| Streamed output tokens inside a window | `benchmarking.aiperf.tokens_in_window(record, start, end)` |
| Server counter totals and gauge statistics | `benchmarking.aiperf.server_total()`, `server_stat()` |
| Completion, error, preemption, saturation and queueing gates | `benchmarking.aiperf.gates(row, min_window_s=5, saturation_ratio=0.9)` |
| JSONL, CSV and Parquet | `benchmarking.artifacts.read_jsonl()`, `write_csv()`, `write_parquet()` |
| AV off/on SPEED-Bench report | `reporting.aiperf_report.main(ReportConfig(...))` |
| Plotting and offline report composition | `reporting/plots.py`, `reporting/model_tabs.py`, `reporting/b200-theme.css` |

## Collect and measure

```python
from benchmarking.aiperf import read_artifacts, steady_window
from benchmarking.artifacts import write_parquet

artifacts = read_artifacts(summary_path)
phase = next(p for p in artifacts.summary["input_config"]["phases"] if p["name"] == "profiling")
metrics = steady_window(artifacts.completed, phase["concurrency"])
write_parquet([{"concurrency": phase["concurrency"], **metrics}], output_path)
```

The window opens at the concurrency-th completion and closes at the last request submission. Throughput counts tokens arriving in `[start, end)`, distributing a request's output tokens equally across its streamed chunks. TPOT and TTFT average requests starting inside the window. ITL averages their chunk gaps. These latency samples and the throughput window have different scopes by design. Empty or nonpositive windows return zero duration and null metrics.

`read_artifacts()` expects sibling `server_metrics_export.json` and `profile_export.jsonl` files. Selecting a phase filters request records and selects its summary when separate phase summaries exist. Server metrics remain whole-cell exports. `server_total()` sums series, `server_stat()` takes the maximum series statistic, neither silently treats missing metrics as zero.

`gates()` requires `concurrency`, `requests`, `requests_target`, `cancelled`, `error_rate`, `preemptions`, `server_running_p50`, `server_waiting_p50` and `window_s`. It returns `valid` and comma-separated `failed_gates`. The default thresholds preserve the existing 5-second SPEED-Bench report policy. Experiments may choose different thresholds explicitly.

Artifact writers expect an existing output directory and nonempty rows. CSV uses the first row's fields. Parquet preserves nested fields through DuckDB JSON inference and uses a temporary staging directory, leaving existing JSONL artifacts intact. Query the resulting Parquet directly with the DuckDB CLI.

## Reports

Use `experiments/qwen_dflash_av_60229/build_report.py` or `experiments/xpress_av_65175/build_report.py` as configuration examples. Their existing CLI options remain available. `ReportConfig` supplies the title, subtitle, speculative width, download prefix, public path substitutions, provenance wording and optional summary fields. Mode suffixes are passed explicitly, there is no shared mutable per-run configuration.

This report builder is for the existing fixed-request SPEED-Bench AV off/on protocol. Its layout and explanatory text describe that protocol. Use shared metric and plotting helpers directly for experiments with different protocols. Read `reporting/AGENTS.md` before editing reports.

When deploying remotely, preserve the package layout by copying `benchmarking/*.py`, `reporting/*.py` and `reporting/b200-theme.css` alongside the experiment tree, or use a complete checkout. The Qwen `make report` target does this. Generic metric ingestion and artifact exports do not import Matplotlib.

## Extend and verify

Add generic operations here and keep experiment-specific decisions at the call site. Prefer small functions and explicit policies. The current layer does not provide remote scheduling, server lifecycle, workload planning, statistical comparisons or profiler ingestion. Search existing experiments before implementing these and extract shared code when adding a second consumer.

Run `~/.venv/bin/python -m unittest discover -s benchmarking/tests`. For migrations, also compare metrics against saved artifacts and rebuild the affected report into its ignored `report/` directory. No model download or GPU startup is needed.
