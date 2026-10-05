# Reporting instructions for agents

## Reuse first

- Read the existing experiment's generator and template before changing its report.
- Use `plots.py` for series plotting, CSV exports, and PNG/SVG exports. Keep experiment-specific metric parsing in its analysis directory.
- `b200-theme.css` contains the shared rules copied from the original B200 report. Reuse it rather than inventing another theme. Embed it in generated HTML so reports open offline.
- Do not add KPI cards, chart tabs, commentary, controls, or extra metrics unless the user asks for them. Preserve the user's chosen layout and colors.

## Plot helpers

Use Matplotlib's `Agg` backend before importing `reporting.plots`. Existing scripts add the repository root to `sys.path` so they run directly without installation.

- `plot_series(ax, points, x, y, **style)` plots rows in their supplied order. Sort by concurrency first when that is the intended sequence.
- `aggregate_concurrency(rows, metrics)` averages repeated runs. Call it separately for each method; it does not group by method itself.
- `save_figure(fig, stem)` writes PNG and SVG and closes the figure.
- `write_csv(rows, path)` uses the first row's keys as columns.
- `is_pareto(point, rows, x, y)` maximizes both metrics. Computing membership does not mean the report should draw a frontier line.

Keep units and aggregation explicit. Reciprocal mean TPOT, mean reciprocal TPOT, median TPOT, and p90 TPOT are different metrics. Histogram bucket bounds are not repeated-run error bars.

## Verify the delivered file

1. Rebuild the actual output directory after every source change. Editing generated HTML alone is not a lasting fix.
2. Open that HTML in a browser and inspect a fresh screenshot, including headings, labels, legends, and every table. Check the user-requested layout, not merely whether the page loads.
3. Compare table values and ordering with the source data. Check that text is readable and labels do not overlap.
4. When the user supplies a screenshot that differs from yours, investigate the viewer's styles. Do not claim their screenshot is wrong. Root-level colors may not carry into an embedded preview; the original theme sets the body color explicitly.
5. Verify the rebuilt file before reporting completion. A source-code color search alone is insufficient.

Use the uv-managed `~/.venv/bin/python`. Plotting needs no GPU reservation. Run checks appropriate to the change; avoid adding tests that merely repeat formatting code.

## Keep Git small

Commit analysis code, templates, shared styles, and documentation. Keep raw data, copied model/benchmark inputs, generated HTML, plots, CSV/JSON results, and caches ignored. Preserve local artifacts when adding ignore rules. Do not add screenshots or copied reference HTML to a code change. Existing tracked artifacts need an explicit decision before removing them from Git.
