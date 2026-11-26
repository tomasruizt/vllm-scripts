"""Throughput-interactivity (Pareto) curve of the vLLM AgentX runs against a reference CSV (e.g. SGLang numbers)."""

import argparse
import json
import re
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reporting.plots import plot_series, save_figure

HERE = Path(__file__).parent
NUM_GPUS = 4
REFERENCE_COLORS = ["#eb6834", "#9a9893"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Run ids under results/, e.g. full-c1-1")
    parser.add_argument("--reference", action="append", default=[], metavar="LABEL=CSV",
                        help="Reference curve from a CSV with concurrency,throughput_per_gpu,interactivity. Repeatable.")
    parser.add_argument("--label", default="vLLM on 4x B200")
    parser.add_argument("--output", type=Path, default=HERE / "plots/pareto")
    args = parser.parse_args()
    ours = sorted((load_run(HERE / "results" / run) for run in args.runs), key=lambda row: row["concurrency"])
    references = [(label, load_reference(Path(path))) for label, path in (ref.split("=", 1) for ref in args.reference)]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(9.5, 5.5), layout="constrained")
    labelled: set[int] = set()  # a later reference only labels the concurrencies that earlier ones lack, to avoid clutter
    for (label, points), color in zip(references, REFERENCE_COLORS):
        draw_series(ax, points, label, color, "s", below=True, skip=frozenset(labelled))
        labelled.update(row["concurrency"] for row in points)
    draw_series(ax, ours, args.label, "#2a78d6", "o", below=False, skip=frozenset())
    rows = ours + [row for _, points in references for row in points]
    ax.set(title="GLM-5.3-Flash AgentX · FP8, TP4, MTP 5 with forced AL 3.60 · 60 min per point",
           xlabel="Interactivity: 1000 / mean ITL (tok/s/user)", ylabel="Throughput per GPU, input + output (tok/s)",
           xlim=(0, max(row["interactivity"] for row in rows) * 1.12), ylim=(0, max(row["throughput"] for row in rows) * 1.12))
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.supxlabel("one run per point · host KV offloading on from c40, off below", fontsize=9, color="#52514e")
    save_figure(fig, args.output)


def load_run(root: Path) -> dict:
    summary = json.loads((root / "aiperf_artifacts/profile_export_aiperf.json").read_text())
    tokens_per_s = summary["input_token_throughput"]["avg"] + summary["output_token_throughput"]["avg"]
    concurrency = int(re.search(r"-c(\d+)-", root.name).group(1))  # run ids look like full-c16-1
    return point(concurrency, tokens_per_s / NUM_GPUS, 1000 / summary["inter_token_latency"]["avg"])


def load_reference(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()[1:]
    return [point(*(float(value) for value in line.split(","))) for line in lines]


def point(concurrency: float, throughput: float, interactivity: float) -> dict:
    return {"concurrency": int(concurrency), "throughput": throughput, "interactivity": interactivity}


def draw_series(ax, points: list[dict], label: str, color: str, marker: str, below: bool, skip: frozenset[int]) -> None:
    plot_series(ax, points, "interactivity", "throughput", label=label, color=color, marker=marker, linewidth=2)
    for row in (row for row in points if row["concurrency"] not in skip):
        ax.annotate(f"c{row['concurrency']}", (row["interactivity"], row["throughput"]), xytext=(-7, -7) if below else (7, 7),
                    textcoords="offset points", fontsize=9, color="#52514e", ha="right" if below else "left",
                    va="top" if below else "bottom")


if __name__ == "__main__":
    main()
