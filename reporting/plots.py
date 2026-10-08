"""Plotting and export primitives shared by benchmark experiments."""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


def aggregate_concurrency(rows, metrics):
    """Average repeated runs at each concurrency, without changing the metric."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["concurrency"]].append(row)
    return [
        {
            "concurrency": concurrency,
            **{key: mean(row[key] for row in runs) for key in metrics},
        }
        for concurrency, runs in sorted(grouped.items())
    ]


def plot_series(ax, points, x, y, **style):
    """Connect points in their supplied order; callers choose their ordering."""
    return ax.plot([row[x] for row in points], [row[y] for row in points], **style)


def scatter_series(ax, points, x, y, **style):
    """Plot individual observations without aggregating them."""
    return ax.scatter([row[x] for row in points], [row[y] for row in points], **style)


def is_pareto(point, rows, x, y):
    """Return whether a measured point is non-dominated when maximizing x and y."""
    return not any(
        other[x] >= point[x]
        and other[y] >= point[y]
        and (other[x] > point[x] or other[y] > point[y])
        for other in rows
    )


def write_csv(rows, path):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, stem, dpi=180):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(stem.with_suffix("." + extension), dpi=dpi)
    plt.close(fig)
