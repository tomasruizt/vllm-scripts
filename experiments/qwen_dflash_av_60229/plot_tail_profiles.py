"""Plot raw startup verification costs with and without valid mixed tails."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    root = Path("results/qwen-av-tail-profile-20261019")
    with (root / "curves.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for arm, label, color in (("slot_only", "State slots fixed", "#b44b36"),
                              ("mixed", "State slots + mixed tail", "#16798b")):
        points = [(int(row["tokens"]), float(row["gpu_ms"])) for row in rows if row["arm"] == arm]
        for axis, limit in zip(axes, (1024, 16384)):
            selected = [(tokens, ms) for tokens, ms in points if tokens <= limit]
            axis.plot([p[0] for p in selected], [p[1] for p in selected],
                      ".-", label=label, color=color, markersize=3)
    for axis in axes:
        axis.set(xlabel="Scheduled target tokens", ylabel="Raw target cost (ms)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    axes[0].set_title("Decode graph range")
    axes[1].set_title("Entire startup profile")
    axes[1].axvline(4096, color="gray", linestyle=":", linewidth=1)
    figure.suptitle("Qwen3.5-35B-A3B / DFlash k=7 / B300 / context 1024\nNative startup medians before clamping")
    for extension in ("png", "svg"):
        figure.savefig(root / f"mixed-tail-cost-curves.{extension}", dpi=180)


if __name__ == "__main__":
    main()
