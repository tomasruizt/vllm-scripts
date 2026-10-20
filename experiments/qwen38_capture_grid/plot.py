"""Plot the saved capture-grid throughput and reciprocal mean TPOT."""

import argparse
import csv
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qwen38-plot-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/qwen38-plot-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reporting.plots import aggregate_concurrency, plot_series, save_figure, scatter_series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent / "results/qwen38-capture-grid-20261019")
    parser.add_argument("--recipe", action="store_true")
    args = parser.parse_args()
    root = args.root
    with (root / "plot.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(9.5, 5.5), layout="constrained")
    styles = [("Default", "#64748b", "o"), ("Up to 2K", "#d97706", "s"), ("Up to 4K", "#08916d", "D")]
    for label, color, marker in styles:
        points = [dict(interactivity=float(row["interactivity"]), throughput=float(row["throughput"]), concurrency=int(row["concurrency"])) for row in rows if row["variant"] == label]
        means = aggregate_concurrency(points, ["interactivity", "throughput"])
        scatter_series(ax, points, "interactivity", "throughput", color=color, marker=marker, alpha=0.45, s=48, edgecolors="none", zorder=3)
        plot_series(ax, means, "interactivity", "throughput", label=label, color=color, marker=marker, linewidth=2, markersize=5)
        for point in means:
            offset = annotation_offset(label, point["concurrency"])
            if args.recipe and label == "Default" and point["concurrency"] == 256:
                offset = (12, -30)
            ax.annotate(f"c{point['concurrency']}", (point["interactivity"], point["throughput"]), xytext=offset, textcoords="offset points", fontsize=10, color=color)
    ax.set(title="Qwen3.8-Flash-Next · CUDA graph capture grid", xlabel="Interactivity: 1 / mean request TPOT (tok/s/user)", ylabel="Steady-state output throughput (tok/s)", xlim=(0, max(float(row["interactivity"]) for row in rows) * 1.15), ylim=(0, max(float(row["throughput"]) for row in rows) * 1.15))
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    runs = len(rows) // 18
    footnote = f"4 × B300 · FP8 recipe · MTP3 · dots: individual runs · lines: means of {runs} runs" if args.recipe else "4 × B300 · BF16 weights · MTP3 · single run per point"
    fig.supxlabel(footnote, fontsize=9, color="#64748b")
    save_figure(fig, root / "throughput-interactivity-expanded")


def annotation_offset(label: str, concurrency: int) -> tuple[int, int]:
    if concurrency == 1:
        return {"Default": (-34, 12), "Up to 2K": (-12, 30), "Up to 4K": (10, 12)}[label]
    if label == "Up to 2K":
        return (-30, -18)
    if label == "Default" and concurrency == 8:
        return (8, -18)
    return (8, 8)


if __name__ == "__main__":
    main()
