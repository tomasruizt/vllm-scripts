"""Step time vs requests and vs tokens per step, AV on and off, from the av-trace "step" records."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qwen-av-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODES = {"offdiag": ("AV off", "#d97706"), "ondiag": ("AV on", "#08916d")}
MAX_GAP_MS = 150  # longer gaps are idle time between cells, not steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    steps = {mode: load_steps(args.run / f"{mode}/repeat-1/av-trace.jsonl") for mode in MODES}
    write_csv(args.output / "step-shapes.csv", steps)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, layout="constrained")
    for ax, (key, label) in zip(axes, (("num_reqs", "requests in the step"), ("num_tokens", "tokens in the step"))):
        draw_panel(ax, steps, key, label)
    axes[0].set_ylabel("step time (ms)")
    axes[1].legend(frameon=False, loc="upper right", markerscale=4)
    fig.suptitle("Step time per executed step, c8 to c128, default CUDA graph sizes", x=0.01, ha="left", fontsize=11)
    for ext in ("png", "svg"):
        fig.savefig(args.output / f"step-shapes.{ext}", dpi=160)
    print(args.output / "step-shapes.png")


def draw_panel(ax: plt.Axes, steps: dict, key: str, label: str) -> None:
    for mode, (name, color) in MODES.items():
        x = np.array([s[key] for s in steps[mode]])
        y = np.array([s["ms"] for s in steps[mode]])
        ax.scatter(x, y, s=3, alpha=0.15, color=color, linewidths=0, label=name, rasterized=True)
    ax.set(xlabel=label, ylim=(0, 120))
    if key == "num_tokens":
        ax.set_xlim(0, 5000)  # beyond: initial-burst steps at cell start (budget-filling prefills)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def load_steps(trace: Path) -> list[dict]:
    """Step records with ms = time until the next step record."""
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    steps = [r for r in records if r["kind"] == "step"]
    rows = [dict(r, ms=(nxt["time_ns"] - r["time_ns"]) / 1e6) for r, nxt in zip(steps, steps[1:])]
    return [r for r in rows if r["ms"] <= MAX_GAP_MS]


def write_csv(path: Path, steps: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["mode,num_reqs,num_tokens,num_draft_tokens,prefill_tokens,cg_mode,step_ms"]
    for mode, rows in steps.items():
        lines += [f"{MODES[mode][0]},{r['num_reqs']},{r['num_tokens']},{r['num_draft_tokens']},{r['prefill_tokens']},{r['cg_mode']},{r['ms']:.3f}"
                  for r in rows]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
