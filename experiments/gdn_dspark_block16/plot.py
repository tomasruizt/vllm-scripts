"""Plot measured throughput against inverse request TPOT."""

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qwen-av-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reporting.plots import (
    aggregate_concurrency,
    plot_series,
    save_figure,
    scatter_series,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Qwen3.6-27B · 2× B300, TP2")
    args = parser.parse_args()
    with args.source.open() as stream:
        rows = [
            dict(
                row,
                concurrency=int(row["concurrency"]),
                throughput=float(row["throughput"]),
                inverse_tpot=1000 / float(row["mean_tpot_ms"]),
            )
            for row in csv.DictReader(stream)
            if row["valid"].lower() == "true"
        ]
    draw(rows, args.output, args.title)


def draw(rows: list[dict], output: Path, title: str) -> None:
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"}
    )
    fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
    styles = [
        ("baseline", "No SD", "#b45309", "s", (7, 8)),
        ("fixed", "Fixed K=15", "#64748b", "o", (7, -14)),
        ("adaptive", "Adaptive K=15", "#138b83", "D", (7, 8)),
    ]
    for mode, label, color, marker, offset in styles:
        runs = [row for row in rows if row["mode"] == mode]
        if not runs:
            continue
        means = aggregate_concurrency(runs, ["inverse_tpot", "throughput"])
        scatter_series(
            ax,
            runs,
            "inverse_tpot",
            "throughput",
            color=color,
            marker=marker,
            alpha=0.35,
            s=45,
        )
        plot_series(
            ax,
            means,
            "inverse_tpot",
            "throughput",
            color=color,
            marker=marker,
            linewidth=2,
            label=label,
        )
        for point in means:
            ax.annotate(
                f"c{point['concurrency']}",
                (point["inverse_tpot"], point["throughput"]),
                xytext=(7, 8) if point["concurrency"] == 1 else offset,
                textcoords="offset points",
                fontsize=9,
                color=color,
            )
    ax.set(
        xlabel="1 / mean request TPOT (tok/s/user, higher is better)",
        ylabel="Steady-state output throughput (tok/s)",
        xlim=(0, max(row["inverse_tpot"] for row in rows) * 1.14),
        ylim=(0, max(row["throughput"] for row in rows) * 1.15),
        title=title,
    )
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.supxlabel(
        "Faint points: individual runs. Lines: means of 3 repeats.", fontsize=9
    )
    save_figure(fig, output)


if __name__ == "__main__":
    main()
