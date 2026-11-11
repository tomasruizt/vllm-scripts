"""Throughput-interactivity (Pareto) curve of main vs this PR for one speculation config, from a summarized run."""

import argparse
import os
from pathlib import Path
from statistics import median
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/topk-topp-plot-cache")
import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reporting.plots import aggregate_concurrency, plot_series, save_figure, scatter_series

SERIES = [("main", "main", "#2a78d6", "o"), ("this PR", "pr", "#eb6834", "s")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="results/<RUN_ID> with metrics.parquet")
    parser.add_argument("--spec", default="k7", help="variant suffix: k7, k8 (DSpark) or mtp3")
    parser.add_argument("--gpu", default="B300")
    parser.add_argument("--model", default="Qwen3.6-35B-A3B")
    parser.add_argument("--variants", default=".*", help="Regex over variant names, e.g. to select server sessions")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "plots/pareto")
    parser.add_argument("--include-invalid", action="store_true", help="Also plot cells that failed a gate (e.g. host_quiet)")
    parser.add_argument("--labels", nargs=2, default=["main", "this PR"], help="Legend labels of main and the PR")
    args = parser.parse_args()
    rows = load_rows(args.root / "metrics.parquet", args.spec, args.variants, args.include_invalid)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(9.5, 5.5), layout="constrained")
    for label, (_, arm, color, marker) in zip(args.labels, SERIES):
        draw_series(ax, [row for row in rows if row["arm"] == arm], label, color, marker, label_points=arm == "pr")
    ax.set(title=f"{args.model} + {spec_title(args.spec)} on {args.gpu} · top-k/top-p masking",
           xlabel="Interactivity: 1 / mean request TPOT (tok/s/user)", ylabel="Steady-state output throughput (tok/s)",
           xlim=(0, max(row["interactivity"] for row in rows) * 1.12), ylim=(0, max(row["throughput"] for row in rows) * 1.12))
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    concurrencies = sorted({row["concurrency"] for row in rows})
    points = "all points" if args.include_invalid else "valid points only"
    fig.supxlabel(f"dots: individual runs · lines: through the medians · {points} (c{concurrencies[0]} to c{concurrencies[-1]})",
                  fontsize=9, color="#52514e")
    save_figure(fig, args.output / f"{args.model.split("-")[0].lower().replace(".", "")}-{args.gpu.lower()}-{args.spec}")


def spec_title(spec: str) -> str:
    return f"DSpark K={spec[1:]}" if spec.startswith("k") else spec.upper()


def load_rows(parquet: Path, spec: str, variants: str, include_invalid: bool) -> list[dict]:
    valid = "true" if include_invalid else "valid"
    query = f"""
        select case when variant like 'main%' then 'main' else 'pr' end arm, concurrency, throughput, 1000 / mean_tpot_ms interactivity
        from '{parquet}' where {valid} and regexp_matches(variant, '^(main|pr|hier)') and regexp_matches(variant, '-{spec}(\\.|$)') and regexp_matches(variant, '{variants}')"""
    columns = ["arm", "concurrency", "throughput", "interactivity"]
    return [dict(zip(columns, row)) for row in duckdb.sql(query).fetchall()]


def draw_series(ax, points: list[dict], label: str, color: str, marker: str, label_points: bool) -> None:
    medians = aggregate_concurrency(points, ["interactivity", "throughput"], statistic=median)
    scatter_series(ax, points, "interactivity", "throughput", color=color, marker=marker, alpha=0.45, s=40,
                   edgecolors="none", zorder=3)
    plot_series(ax, medians, "interactivity", "throughput", label=label, color=color, marker="", linewidth=2,
                zorder=2)
    if not label_points:  # label each concurrency once, above-left of the PR curve
        return
    for point in medians:
        ax.annotate(f"c{point['concurrency']}", (point["interactivity"], point["throughput"]), xytext=(-7, 7),
                    textcoords="offset points", fontsize=9, color="#52514e", ha="right", va="bottom")


if __name__ == "__main__":
    main()
