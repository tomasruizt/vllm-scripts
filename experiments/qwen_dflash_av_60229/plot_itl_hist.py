"""Histogram of per-chunk ITL (inter-chunk latency), AV off vs on, one panel per concurrency."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qwen-av-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODES = {"off": "#d97706", "on": "#08916d"}
LEVELS = (8, 32, 64, 128)
BINS = np.arange(0, 121, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(len(LEVELS), 1, figsize=(9, 2.3 * len(LEVELS)), sharex=True, layout="constrained")
    for ax, c in zip(axes, LEVELS):
        draw_panel(ax, args.source, c)
    fig.suptitle("ITL distribution, AV off vs on (3 repeats pooled)", x=0.01, ha="left", fontsize=11)
    axes[-1].set_xlabel("ITL: time between streamed chunks (ms)")
    axes[0].legend(frameon=False, loc="upper right")
    args.output.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(args.output / f"itl-histogram.{ext}", dpi=160)
    print(args.output / "itl-histogram.png")


def draw_panel(ax: plt.Axes, source: Path, c: int) -> None:
    for mode, color in MODES.items():
        itls = chunk_latencies(source, mode, c)
        ax.hist(np.clip(itls, BINS[0], BINS[-1]), bins=BINS, density=True, histtype="step", color=color, linewidth=1.6,
                label=f"AV {mode}")
        ax.axvline(itls.mean(), color=color, linewidth=1, linestyle=":")
        ax.annotate(f"mean {itls.mean():.1f}", (itls.mean(), 1), xycoords=("data", "axes fraction"),
                    xytext=(3, -12 if mode == "off" else -24), textcoords="offset points", fontsize=8, color="#374151")
    ax.set_title(f"concurrency {c}", loc="left", fontsize=10)
    ax.set_ylabel("density")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right", "left"]].set_visible(False)


def chunk_latencies(source: Path, mode: str, c: int) -> np.ndarray:
    """All inter-chunk latencies (ms) of all requests in the main c-level cells of one mode."""
    paths = sorted(source.glob(f"{mode}/repeat-*/artifacts-c{c}/profile_export.jsonl"))
    records = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
    return np.array([v for r in records for v in r["metrics"].get("inter_chunk_latency", {}).get("value", [])])


if __name__ == "__main__":
    main()
