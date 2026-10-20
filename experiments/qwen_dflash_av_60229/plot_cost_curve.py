"""Profiled AV verify cost vs measured decode-only step times, AV on and off, from av-trace.jsonl files."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qwen-av-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LIMIT = 1024  # CUDA graph capture limit
K = 7  # draft tokens per request
LAG = 2  # output records trail AV budget records by two steps (async scheduling)
SLOW_MS = 60  # steps above this contain a prefill
# Measured median step time with one prefill (c8 to c128, AV off), from client-side step reconstruction.
PREFILL = [(1160, 78.2), (1390, 81.0), (1765, 83.8), (2525, 90.4)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="run dir with ondiag/ and offdiag/ traces")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    on = load(args.run / "ondiag/repeat-1/av-trace.jsonl")
    off = load(args.run / "offdiag/repeat-1/av-trace.jsonl")
    on_points = av_on_points(on)
    off_points = av_off_points(off)
    write_csv(args.output / "decode-steps.csv", on_points, off_points)
    draw(np.array(full_curve(on)), on_points, off_points, args.output)


def draw(curve: np.ndarray, on_points: list, off_points: list, output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    xs = np.arange(1, 2701)
    ax.plot(xs, cost_table(curve, xs), color="#4b5563", linewidth=2, label="cost table AV uses (clamped, interpolated)")
    shown = curve[curve[:, 0] <= 2700]
    ax.scatter(shown[:, 0], shown[:, 1], s=12, color="#9ca3af", zorder=3, label="profiled dummy batches (median)")
    scatter(ax, off_points, "#d97706", "o", "measured decode-only step, AV off (one per request count)")
    scatter(ax, on_points, "#08916d", "D", "measured decode-only step, AV on (binned by size)")
    scatter(ax, PREFILL, "#d97706", "s", "measured step with one prefill, AV off")
    ax.axvline(LIMIT, color="#9ca3af", linestyle=":", linewidth=1)
    ax.annotate("CUDA graph limit", (LIMIT, 95), xytext=(4, 0), textcoords="offset points", fontsize=8, color="#374151")
    ax.set(xlabel="tokens in the step", ylabel="step time (ms)", xlim=(0, 2700), ylim=(0, 100))
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    for ext in ("png", "svg"):
        fig.savefig(output / f"cost-curve.{ext}", dpi=160)
    print(output / "cost-curve.png")


def av_off_points(records: list) -> list:
    """One point per request count: x = requests x (1 + K) tokens, y = median decode-only step time."""
    steps = [(r["num_reqs"], gap) for r, gap in output_steps(records) if gap <= SLOW_MS]
    groups: dict[int, list[float]] = {}
    for num_reqs, gap in steps:
        groups.setdefault(num_reqs, []).append(gap)
    return [(n * (1 + K), float(np.median(g))) for n, g in sorted(groups.items()) if len(g) >= 30]


def av_on_points(records: list) -> list:
    """Decode-only steps binned by tokens in the step (bonus + admitted drafts), median step time per bin."""
    budgets = [r for r in records if r["kind"] == "budget"]
    outputs = output_steps(records)
    pairs = [(b["non_draft_total"] + b["budget"], outputs[i + LAG][1])
             for i, b in enumerate(budgets) if i + LAG < len(outputs) and not b["has_prefill"]]
    tokens = np.array([t for t, _ in pairs])
    gaps = np.array([g for _, g in pairs])
    edges = np.arange(0, 1025 + 64, 64)
    points = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (tokens >= lo) & (tokens < hi)
        if in_bin.sum() >= 30:
            points.append((float(np.median(tokens[in_bin])), float(np.median(gaps[in_bin]))))
    return points


def output_steps(records: list) -> list:
    """(output record, ms since the previous output) for every output after the first."""
    outputs = [r for r in records if r["kind"] == "output"]
    return [(r, (r["time_ns"] - prev["time_ns"]) / 1e6) for prev, r in zip(outputs, outputs[1:])]


def load(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines()]


def full_curve(records: list) -> list:
    """The startup verify curve with the most profiled sizes."""
    return max((r["verify_curve"] for r in records if r["kind"] == "costs"), key=len)


def cost_table(curve: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Same construction as build_cost_tables_from_curves: clamp to non-decreasing, step below the limit, interpolate above."""
    sizes = curve[:, 0]
    ms = np.maximum.accumulate(curve[:, 1])
    table = ms[np.minimum(np.searchsorted(sizes, xs), len(sizes) - 1)]
    above = sizes > LIMIT
    table[xs > LIMIT] = np.interp(xs[xs > LIMIT], sizes[above], ms[above])
    return table


def scatter(ax: plt.Axes, points: list, color: str, marker: str, label: str) -> None:
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    ax.scatter(x, y, s=46, color=color, marker=marker, edgecolors="white", linewidths=1, zorder=4, label=label)


def write_csv(path: Path, on_points: list, off_points: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [f"on,{x:.0f},{y:.2f}" for x, y in on_points] + [f"off,{x:.0f},{y:.2f}" for x, y in off_points]
    path.write_text("mode,tokens,step_ms\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
