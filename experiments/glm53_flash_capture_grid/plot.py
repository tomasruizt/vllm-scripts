"""Plot valid GLM capture-grid runs and their mean throughput/interactivity."""
import argparse
import csv
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/glm53-plot-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reporting.plots import aggregate_concurrency, plot_series, save_figure, scatter_series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent / "results/glm53-flash-capture-pilot-20261019")
    root = parser.parse_args().root
    with (root / "plot.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    points = [dict(variant=row["variant"], concurrency=int(row["concurrency"]), throughput=float(row["throughput"]), interactivity=float(row["interactivity"])) for row in rows]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(9.5, 5.5), layout="constrained")
    for label, color, marker in [("Default", "#64748b", "o"), ("Up to 4K", "#08916d", "D")]:
        runs = [point for point in points if point["variant"] == label]
        means = aggregate_concurrency(runs, ["interactivity", "throughput"])
        scatter_series(ax, runs, "interactivity", "throughput", color=color, marker=marker, alpha=0.45, s=60, edgecolors="none", zorder=3)
        plot_series(ax, means, "interactivity", "throughput", label=label, color=color, marker=marker, linewidth=2, markersize=5)
        for point in means:
            ax.annotate(f"c{point['concurrency']}", (point["interactivity"], point["throughput"]), xytext=(9, 9), textcoords="offset points", fontsize=10, color=color)
    ax.set(title="GLM-5.3-Flash · CUDA graph capture grid", xlabel="Interactivity: 1 / mean request TPOT (tok/s/user)", ylabel="Steady-state output throughput (tok/s)", xlim=(0, max(p["interactivity"] for p in points) * 1.17), ylim=(0, max(p["throughput"] for p in points) * 1.17))
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.supxlabel("4 × B300 · FP8 · MTP5 · valid runs only\nc64: 3 runs/variant · c128: 2 runs/variant · dots: individual runs · lines: means", fontsize=9, color="#64748b")
    save_figure(fig, root / "throughput-interactivity")


if __name__ == "__main__":
    main()
