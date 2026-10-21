"""Compare AV cost curves: verify cost vs tokens, optionally drafter cost vs requests.

Defaults reproduce the unfixed base vs final PR startup-profile comparison. Arms are
`path:label:color`, where path is a curve JSON (`draft_curve`, `verify_curve`) or a
"DSpark cost tables" debug log line saved by dump-cost-tables.sh.
"""

import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

ROOT = Path(__file__).resolve().parent / "results"
DEFAULT_ARMS = [
    f"{ROOT}/qwen-gdn-profile-fix-20261019/base.json:Base:#b44b36",
    f"{ROOT}/qwen-av-tail-profile-20261019/mixed.json:PR:#16798b",
]
DEFAULT_TITLE = "Unfixed base vs final PR: Qwen3.5-35B-A3B / DFlash k=7 / B300\nNative startup medians before clamping, context 1024"


def main() -> None:
    args = parse_args()
    panels = 3 if args.draft else 2
    figure, axes = plt.subplots(1, panels, figsize=(5.5 * panels, 4), layout="constrained")
    for spec in args.arm or DEFAULT_ARMS:
        plot_arm(axes, spec, args.draft)
    for axis in axes[:2]:
        axis.set(xlabel="Scheduled target tokens", ylabel=args.verify_label)
    axes[0].set_title("Decode graph range")
    axes[1].set_title("Entire token range")
    if args.draft:
        axes[2].set(xlabel="Requests in batch", ylabel="Drafter cost (ms)", title="Drafter")
    for axis in axes:
        axis.set(ylim=(0, None))
        axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.suptitle(args.title)
    for extension in ("png", "svg"):
        figure.savefig(f"{args.output}.{extension}", dpi=180)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", help="path:label:color, repeatable")
    parser.add_argument("--draft", action="store_true", help="add a drafter cost panel")
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--verify-label", default="Raw target cost (ms)")
    parser.add_argument("--output", default=str(ROOT / "qwen-av-tail-profile-20261019/base-vs-pr-cost-curves"))
    return parser.parse_args()


def plot_arm(axes: list, spec: str, draft: bool) -> None:
    path, label, color = spec.rsplit(":", 2)
    curves = load_curves(Path(path))
    for axis, limit in zip(axes, (1024, 16384)):
        selected = [(tokens, ms) for tokens, ms in curves["verify_curve"] if 1 <= tokens <= limit]
        plot_points(axis, selected, label, color)
    if draft:
        plot_points(axes[2], [(reqs, ms) for reqs, ms in curves["draft_curve"] if reqs >= 1], label, color)


def plot_points(axis: plt.Axes, points: list, label: str, color: str) -> None:
    # Dense lookup tables get a plain line, knot curves get markers.
    marker = ".-" if len(points) < 300 else "-"
    axis.plot([p[0] for p in points], [p[1] for p in points], marker, label=label, color=color, markersize=3)


def load_curves(path: Path) -> dict:
    if path.suffix == ".json":
        return json.loads(path.read_text())
    # Dense lookup tables printed by AV at startup: index = requests / tokens.
    text = path.read_text().split("DSpark cost tables: ", 1)[1]
    tables = [[float(x) for x in re.split(r"[,\s]+", body.strip()) if x]
              for body in re.findall(r"array\(\[(.*?)\]\)", text, re.S)]
    draft, verify = tables
    return dict(draft_curve=list(enumerate(draft)), verify_curve=list(enumerate(verify)))


if __name__ == "__main__":
    main()
