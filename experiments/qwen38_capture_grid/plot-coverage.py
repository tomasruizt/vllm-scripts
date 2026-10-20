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
        values = {int(row["concurrency"]): float(row["graphed_pct"]) for row in rows if row["variant"] == label}
        bars = ax.bar([i + (index - 1) * 0.25 for i in range(len(levels))], [values[c] for c in levels], width=0.24, label=label, color=color)
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
    fig.supxlabel(f"{scope} · counts pooled across {runs} runs · includes filling and draining\nStep coverage, not request or runtime share. Piecewise graphs retain eager sections.", fontsize=9, color="#64748b")
    save_figure(fig, root / f"{stem}-bars")


if __name__ == "__main__":
    main()
