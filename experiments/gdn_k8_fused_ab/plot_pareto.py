"""Throughput-interactivity (Pareto) curve of main vs this PR for one speculation config, from a summarized run."""

import argparse
import os
from pathlib import Path
from statistics import median
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gdn-k8-plot-cache")
import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reporting.plots import aggregate_concurrency, plot_series, save_figure, scatter_series

SERIES = [("main", "main", "#2a78d6", "o"), ("this PR", "splitv", "#eb6834", "s")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="results/<RUN_ID> with metrics.parquet")
    parser.add_argument("--spec", default="k8", help="variant suffix: k7, k8 (DSpark) or mtp3")
    parser.add_argument("--gpu", default="B300")
    parser.add_argument("--variants", default=".*", help="Regex over variant names, e.g. to select server sessions")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "plots/pareto")
    args = parser.parse_args()
    rows = load_rows(args.root / "metrics.parquet", args.spec, args.variants)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(9.5, 5.5), layout="constrained")
    for label, arm, color, marker in SERIES:
        draw_series(ax, [row for row in rows if row["arm"] == arm], label, color, marker)
    ax.set(title=f"Qwen3.6-35B-A3B + {spec_title(args.spec)} on {args.gpu} · fused GDN MTP decode",
           xlabel="Interactivity: 1 / mean request TPOT (tok/s/user)", ylabel="Steady-state output throughput (tok/s)",
           xlim=(0, max(row["interactivity"] for row in rows) * 1.12), ylim=(0, max(row["throughput"] for row in rows) * 1.12))
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    concurrencies = sorted({row["concurrency"] for row in rows})
    fig.supxlabel(f"dots: individual runs · lines: through the medians · valid points only (c{concurrencies[0]} to c{concurrencies[-1]})",
                  fontsize=9, color="#52514e")
    save_figure(fig, args.output / f"{args.gpu.lower()}-{args.spec}")


def spec_title(spec: str) -> str:
    return f"DSpark K={spec[1:]}" if spec.startswith("k") else spec.upper()


def load_rows(parquet: Path, spec: str, variants: str) -> list[dict]:
    query = f"""
        select regexp_extract(variant, '^(main|splitv)') arm, concurrency, throughput, 1000 / mean_tpot_ms interactivity
        from '{parquet}' where valid and regexp_matches(variant, '-{spec}(\\.|$)') and regexp_matches(variant, '{variants}')"""
    columns = ["arm", "concurrency", "throughput", "interactivity"]
    return [dict(zip(columns, row)) for row in duckdb.sql(query).fetchall()]


def draw_series(ax, points: list[dict], label: str, color: str, marker: str) -> None:
    medians = aggregate_concurrency(points, ["interactivity", "throughput"], statistic=median)
    scatter_series(ax, points, "interactivity", "throughput", color=color, marker=marker, alpha=0.45, s=40,
                   edgecolors="none", zorder=3)
    plot_series(ax, medians, "interactivity", "throughput", label=label, color=color, marker="", linewidth=2,
                zorder=2)
    for point in medians:
        below = label == "main"  # main sits below this PR, so its labels go below-right and the PR labels above-left
        ax.annotate(f"c{point['concurrency']}", (point["interactivity"], point["throughput"]), xytext=(7, -7) if below else (-7, 7),
                    textcoords="offset points", fontsize=9, color="#52514e", ha="left" if below else "right", va="top" if below else "bottom")


if __name__ == "__main__":
    main()
