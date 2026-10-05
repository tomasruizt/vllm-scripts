# GLM-5.3 report instructions

Read `../../../reporting/AGENTS.md` first.

## Files and layout

- `build_report.py`: reads archived evaluations and Prometheus snapshots, creates plots and tables, then renders HTML.
- `template.html`: title, one throughput/interactivity chart, and two results tables in tabs: throughput and accuracy/AL. Keep this minimal layout.
- `../../../reporting/b200-theme.css`: the original B200 theme. Do not approximate it or replace it with custom gray/black overrides.
- `sources.json`: optional local mapping of methods to run directories; ignored because paths depend on the machine.
- Default and current output: `../report/`, under `experiments/glm53_av/`.
- Benchmark notes and tables: `../glm-5.3-ep-dp-av.md`.
- Local original theme reference: `../../../reporting/references/b200-dflash-report.html` (ignored).

## Rebuild

From the `vllm-scripts` root, rebuild using archived inputs already in the output directory:

```bash
~/.venv/bin/python experiments/glm53_av/analysis/build_report.py
```

Add `--sources experiments/glm53_av/analysis/sources.json` only when importing completed runs. A sources file maps `baseline`, `avoff`, and `avon` to lists of run directories. Each directory needs `result.json`, its listed evaluations and before/after metric snapshots, client logs, and `run.py`. No model download or server restart is needed for plotting.

The output's `data/` copies are enough to rebuild after `/tmp` is gone. Each run's `provenance.json` records its command and revisions; `data-sha256.json` checks the archived inputs. Keep these files locally, outside Git.

## Data and display rules

- Bsz means total client concurrency across all four DP ranks.
- Throughput table: rows 8, 64, 128, 256; columns no speculation, DSpark AV off, DSpark + AV. Show output tok/s to one decimal.
- Accuracy/AL table: group by method, then bsz. Show accuracy to two decimals with `%`, AL to three decimals, and `N/A` for the baseline.
- These tables must match `experiments/glm53_av/glm-5.3-ep-dp-av.md`. Compare rendered cells, not only numeric arrays.
- Main chart uses `1 / mean per-request TPOT`, from summed Prometheus counter deltas. Validate 1,319 observations per point. The B200 reference uses p90; do not silently substitute that metric.
- No KPI cards, throughput-by-concurrency panel, estimated-p90 tab, or drawn Pareto line/rings. Exported data may retain additional metrics without displaying them.
- Series colors are the original palette: blue `#2563eb`, amber `#d97706`, green `#08916d`.

After rebuilding, visually inspect the actual HTML, switch both table tabs, and check both tables against the Markdown. Throughput is selected initially. Verify that switching tabs shows only the selected table and that keyboard focus remains visible. If a user reports different rendering, inspect the supplied screenshot and embedded-viewer styles before claiming the issue is fixed.
