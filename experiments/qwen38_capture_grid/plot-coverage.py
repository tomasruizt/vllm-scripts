"""Plot execution-step CUDA graph coverage from the recipe comparison."""

import argparse
import csv
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qwen38-plot-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reporting.plots import save_figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill", action="store_true")
    parser.add_argument("--min-max", action="store_true", help="Show per-run means with min–max error bars")
    parser.add_argument("--root", type=Path, default=Path(__file__).parent / "results/qwen38-recipe-grid-20261019")
    args = parser.parse_args()
    root = args.root
    stem = "graph-coverage-prefill" if args.prefill else "graph-coverage"
    with (root / f"{stem}.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    levels = [1, 8, 32, 64, 128, 256]
    styles = [("Default", "#64748b"), ("Up to 2K", "#d97706"), ("Up to 4K", "#08916d")]
    runs = int(rows[0]["runs"])
    assert len(rows) == 18 and all(int(row["runs"]) == runs for row in rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(11, 5.8), layout="constrained")
    for index, (label, color) in enumerate(styles):
        selected = {int(row["concurrency"]): row for row in rows if row["variant"] == label}
        metric = "mean_pct" if args.min_max else "graphed_pct"
        values = {c: float(selected[c][metric]) for c in levels}
        errors = [[values[c] - float(selected[c]["min_pct"]) for c in levels], [float(selected[c]["max_pct"]) - values[c] for c in levels]] if args.min_max else None
        bars = ax.bar([i + (index - 1) * 0.25 for i in range(len(levels))], [values[c] for c in levels], width=0.24, label=label, color=color, yerr=errors, capsize=3, error_kw={"elinewidth": 1, "capthick": 1})
        labels = [f"{values[c]:.1f}".rstrip("0").rstrip(".") + "%" for c in levels]
        ax.bar_label(bars, labels=labels, padding=4, fontsize=9)
    title = "Qwen3.8-Flash-Next · Prefill/mixed-step CUDA graph coverage" if args.prefill else "Qwen3.8-Flash-Next · CUDA graph coverage"
    ylabel = "Prefill/mixed steps using CUDA graphs" if args.prefill else "Execution steps using CUDA graphs"
    ax.set(title=title, xlabel="Concurrency", ylabel=ylabel, ylim=(0, 116), xticks=range(len(levels)), xticklabels=levels, yticks=range(0, 101, 20))
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    scope = "Steps > 4 × concurrency tokens (MTP3 decode-only bound)" if args.prefill else "FULL + PIECEWISE dispatches"
    summary = f"bars: mean of {runs} runs · error bars: min–max" if args.min_max else f"counts pooled across {runs} runs"
    fig.supxlabel(f"{scope} · {summary}\nIncludes filling and draining. Piecewise graphs retain eager sections.", fontsize=9, color="#64748b")
    save_figure(fig, root / f"{stem}-bars")


if __name__ == "__main__":
    main()
