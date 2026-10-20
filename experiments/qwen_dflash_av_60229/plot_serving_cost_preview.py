"""Preview base serving periods against both complete AV cost estimates."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "plots" / "community-gdn-preview"


def main() -> None:
    with (ROOT / "plots/step-shapes.csv").open() as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["num_tokens"]) <= 16384]
    estimates = load_estimates()
    OUTPUT.mkdir(exist_ok=True)
    write_predictions(rows, estimates)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True,
                                gridspec_kw={"width_ratios": [2, 1]}, layout="constrained")
    for axis, lower, upper in ((axes[0], 0, 4096), (axes[1], 4096, 16384)):
        selected = [row for row in rows if lower < int(row["num_tokens"]) <= upper]
        draw_panel(axis, selected, estimates, lower, upper)
    figure.suptitle("AV calibration before clamping")
    figure.supxlabel("Scheduled target tokens")
    for extension in ("png", "svg"):
        figure.savefig(OUTPUT / f"serving-cost-comparison.{extension}", dpi=180)


def draw_panel(axis: object, rows: list[dict], estimates: dict, lower: int, upper: int) -> None:
    tokens = np.array([int(row["num_tokens"]) for row in rows])
    measured = np.array([float(row["step_ms"]) for row in rows])
    requests = np.array([int(row["num_reqs"]) for row in rows])
    mixed = np.array([int(row["prefill_tokens"]) > 0 for row in rows])
    for mask, label, color in (
        (~mixed, "Measured decode-only steps", "#597bc2"),
        (mixed, "Measured mixed batches", "#d19b26"),
    ):
        axis.scatter(tokens[mask], measured[mask], s=6, alpha=0.22, color=color,
                     linewidths=0, rasterized=True, label=label)
    series = [(label, draft[requests] + verify[tokens], color)
              for label, (draft, verify, color) in estimates.items()]
    bins = tokens // (64 if lower == 0 else 512)
    populated = np.unique(bins)
    for label, values, color in series:
        x = [np.median(tokens[bins == value]) for value in populated]
        y = [np.median(values[bins == value]) for value in populated]
        axis.plot(x, y, ".-", color=color, label=label, linewidth=2, markersize=4)
    axis.set(ylabel="Step period / estimated cost (ms)" if lower == 0 else "",
             xlim=(lower, upper), ylim=(0, 200))
    axis.grid(alpha=0.2)
    if lower == 0:
        axis.legend(frameon=False, fontsize=8, loc="upper left", markerscale=3)
    else:
        ticks = np.arange(6000, 16001, 2000)
        axis.set_xticks(ticks, [f"{value // 1000}K" for value in ticks])


def load_estimates() -> dict:
    result = {}
    for label, relative, color in (
        ("Community PR", "qwen-gdn-community-20261019/community.json", "#b44b36"),
        ("Community PR + GDN fixes", "qwen-gdn-community-20261019/gdn.json", "#16798b"),
    ):
        curves = json.loads((ROOT / "results" / relative).read_text())
        draft = raw_table(curves["draft_curve"], 512)
        verify = raw_table(curves["verify_curve"], 16384)
        result[label] = (draft, verify, color)
    return result


def raw_table(curve: list, limit: int) -> np.ndarray:
    # Retain graph-size lookup/interpolation, without the cumulative maximum.
    sizes, costs = np.asarray(curve, dtype=float).T
    tokens = np.arange(limit + 1)
    indices = np.minimum(np.searchsorted(sizes, tokens), len(sizes) - 1)
    values = costs[indices]
    above = sizes > 1024
    if above.any():
        values[tokens > 1024] = np.interp(tokens[tokens > 1024], sizes[above], costs[above])
    return values


def write_predictions(rows: list[dict], estimates: dict) -> None:
    with (OUTPUT / "step-predictions.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*rows[0], *estimates])
        writer.writeheader()
        for row in rows:
            predictions = {label: float(draft[int(row["num_reqs"])] + verify[int(row["num_tokens"])])
                           for label, (draft, verify, _) in estimates.items()}
            writer.writerow({**row, **predictions})


if __name__ == "__main__":
    main()
