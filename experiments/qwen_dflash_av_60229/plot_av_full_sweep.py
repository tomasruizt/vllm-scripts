"""Plot matched AV cost policies using the native AIPerf sweep results."""

import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/qwen-av-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reporting.plots import save_figure


def main() -> None:
    source = Path("results/qwen-av-full-sweep-20261020/analysis/metrics.parquet")
    output = Path("plots/full-curve-sweep")
    output.mkdir(parents=True, exist_ok=True)
    rows = query(f"SELECT mode,concurrency,throughput,1000/mean_tpot_ms interactivity FROM '{source}' WHERE included AND valid")
    medians = query(f"SELECT mode,concurrency,count(*) runs,median(throughput) throughput,1000/median(mean_tpot_ms) interactivity FROM '{source}' WHERE included AND valid GROUP BY mode,concurrency ORDER BY mode,concurrency")
    if len(rows) != 72 or len(medians) != 12 or any(row["runs"] != 6 for row in medians):
        raise RuntimeError("Expected six valid repeats for each cost policy and concurrency")
    fig, ax = plt.subplots(figsize=(9.5, 6.2), layout="constrained")
    draw_policy(ax, rows, medians, "native", "Native AV cost curve", "#d97706", "o")
    draw_policy(ax, rows, medians, "measured", "Measured AV cost curve", "#0f766e", "D")
    label_concurrencies(ax, medians)
    ax.set(xlabel="Interactivity (tok/s/user, 1000 / mean TPOT in ms)",
           ylabel="Steady-state output throughput (tok/s)", xlim=(0, 660), ylim=(0, 20500))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.set_title("Adaptive verification: native versus measured costs\n"
                 "Qwen3.5-35B-A3B + DFlash · 2× B300 (TP2) · k=7", loc="left", pad=16)
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    fig.supxlabel("Faint points: six individual runs per concurrency. Lines: medians. Higher and farther right is better.",
                  fontsize=9)
    save_figure(fig, output / "throughput-interactivity")
    (output / "plotted-values.json").write_text(json.dumps(dict(runs=rows, medians=medians), indent=2))
    print(output / "throughput-interactivity.png")


def draw_policy(ax: object, rows: list[dict], medians: list[dict], mode: str,
                label: str, color: str, marker: str) -> None:
    samples = [row for row in rows if row["mode"] == mode]
    centers = [row for row in medians if row["mode"] == mode]
    ax.scatter([r["interactivity"] for r in samples], [r["throughput"] for r in samples],
               color=color, marker=marker, alpha=.25, s=28, edgecolors="none", zorder=2)
    ax.plot([r["interactivity"] for r in centers], [r["throughput"] for r in centers],
            color=color, marker=marker, markersize=6, linewidth=1.8, label=label, zorder=3)


def label_concurrencies(ax: object, medians: list[dict]) -> None:
    for c in (1, 8, 32, 64, 128, 256):
        pair = [r for r in medians if r["concurrency"] == c]
        x = sum(r["interactivity"] for r in pair) / 2
        y = sum(r["throughput"] for r in pair) / 2
        offset = (12, 10) if c == 256 else (10, 7)
        ax.annotate(f"c{c}", (x, y), xytext=offset, textcoords="offset points", fontsize=10,
                    color="#374151", bbox=dict(facecolor="white", edgecolor="none", alpha=.75, pad=1))


def query(sql: str) -> list[dict]:
    result = subprocess.run(["duckdb", "-json", "-c", sql], text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


if __name__ == "__main__":
    main()
