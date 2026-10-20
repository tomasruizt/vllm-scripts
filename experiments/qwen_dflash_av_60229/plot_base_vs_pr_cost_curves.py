"""Compare the unfixed base directly with the final PR startup profile."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


def main() -> None:
    root = Path(__file__).resolve().parent / "results"
    arms = (
        ("qwen-gdn-profile-fix-20261019/base.json", "Base", "#b44b36"),
        ("qwen-av-tail-profile-20261019/mixed.json", "PR", "#16798b"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for source, label, color in arms:
        points = json.loads((root / source).read_text())["verify_curve"]
        for axis, limit in zip(axes, (1024, 16384)):
            selected = [(tokens, ms) for tokens, ms in points if tokens <= limit]
            axis.plot([p[0] for p in selected], [p[1] for p in selected],
                      ".-", label=label, color=color, markersize=3)
    for axis in axes:
        axis.set(xlabel="Scheduled target tokens", ylabel="Raw target cost (ms)", ylim=(0, None))
        axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    axes[0].set_title("Decode graph range")
    axes[1].set_title("Entire startup profile")
    figure.suptitle("Unfixed base vs final PR: Qwen3.5-35B-A3B / DFlash k=7 / B300\nNative startup medians before clamping, context 1024")
    for extension in ("png", "svg"):
        figure.savefig(root / "qwen-av-tail-profile-20261019" / f"base-vs-pr-cost-curves.{extension}", dpi=180)


if __name__ == "__main__":
    main()
