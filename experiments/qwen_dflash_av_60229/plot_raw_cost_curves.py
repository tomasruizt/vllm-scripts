"""Plot native AV startup verify costs before clamping, from DuckDB's CSV export."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    root = Path("results/qwen-gdn-profile-fix-20261019")
    with (root / "raw-costs.csv").open() as handle:
        rows = [row for row in csv.DictReader(handle) if row["scope"] == "verify"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for arm, color in (("base", "#b44b36"), ("fixed", "#16798b")):
        points = [(int(row["batch_size"]), float(row["gpu_ms"])) for row in rows if row["arm"] == arm]
        for axis, limit in zip(axes, (1024, 16384)):
            selected = [(tokens, ms) for tokens, ms in points if tokens <= limit]
            axis.plot([p[0] for p in selected], [p[1] for p in selected],
                      ".-", label=arm, color=color, markersize=3)
    for axis in axes:
        axis.set(xlabel="Scheduled target tokens", ylabel="Raw target cost (ms)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    axes[0].set_title("Decode graph range")
    axes[0].axvline(512, color="gray", linestyle=":", linewidth=1)
    axes[0].text(512, 0.97, "512 × 1", transform=axes[0].get_xaxis_transform(), ha="right", va="top")
    axes[0].text(1024, 0.97, "512 × 2", transform=axes[0].get_xaxis_transform(), ha="right", va="top")
    axes[1].set_title("Entire startup profile, including eager tail")
    figure.suptitle("Qwen3.5-35B-A3B / DFlash k=7 / B300 / context 1024\nNative startup medians before monotonic clamping")
    for extension in ("png", "svg"):
        figure.savefig(root / f"raw-cost-curves.{extension}", dpi=180)


if __name__ == "__main__":
    main()
