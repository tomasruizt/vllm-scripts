"""Plotting and export primitives shared by benchmark experiments."""

from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from benchmarking.artifacts import write_csv


def aggregate_concurrency(rows, metrics, statistic=mean):
    """Aggregate repeated runs at each concurrency (mean by default), without changing the metric."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["concurrency"]].append(row)
    return [
        {
            "concurrency": concurrency,
            **{key: statistic([row[key] for row in runs]) for key in metrics},
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


def save_figure(fig, stem, dpi=180, svg=False):
    """Write <stem>.png, and <stem>.svg only with svg=True (e.g. for reports that embed it)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg") if svg else ("png",):
        fig.savefig(stem.with_suffix("." + extension), dpi=dpi)
    plt.close(fig)
