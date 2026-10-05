#!/usr/bin/env python3
"""Summarize a vllm bench sweep serve experiment and plot its tradeoff."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from reporting.plots import (
    aggregate_concurrency,
    plot_series,
    save_figure,
    write_csv,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "qwen3_32b_fp8_tp2_complete"


def main() -> None:
    rows = load_rows()
    if not rows:
        raise SystemExit(f"No completed benchmark runs under {RESULTS}")
    write_csv(rows, HERE / "summary.csv")
    plot_interactivity(rows)
    print(f"Wrote {len(rows)} rows to {HERE / 'summary.csv'}")


def load_rows() -> list[dict]:
    rows = []
    paths = list(RESULTS.glob("SERVE--*-BENCH--*/run=*.json"))
    for path in paths:
        label, concurrency = path.parent.name.removeprefix("SERVE--").split("-BENCH--")
        data = json.loads(path.read_text())
        if data.get("completed", 0) == 0:
            continue
        rows.append(
            {
                "arm": label,
                "concurrency": int(concurrency.removeprefix("c")),
                "run": int(path.stem.removeprefix("run=")),
                "completed": data.get("completed"),
                "output_throughput": data["output_throughput"],
                "median_tpot_ms": data["median_tpot_ms"],
                "interactivity_tok_s_user": 1000 / data["median_tpot_ms"],
                "p99_tpot_ms": data.get("p99_tpot_ms"),
                "median_ttft_ms": data.get("median_ttft_ms"),
                "spec_decode_acceptance_length": data.get(
                    "spec_decode_acceptance_length"
                ),
                "spec_decode_draft_tokens": data.get("spec_decode_draft_tokens"),
                "spec_decode_num_drafts": data.get("spec_decode_num_drafts"),
                "result_path": str(path),
            }
        )
    return sorted(rows, key=lambda x: (x["arm"], x["concurrency"], x["run"]))


def plot_interactivity(rows: list[dict]) -> None:
    fig, (fixed_ax, adaptive_ax) = plt.subplots(
        1, 2, figsize=(14, 6), sharex=True, sharey=True, constrained_layout=True
    )
    ks = [int(arm.split("_k")[1]) for arm in {row["arm"] for row in rows}]
    min_k, max_k = min(ks), max(ks)
    for arm in sorted({row["arm"] for row in rows}):
        if arm == "adaptive_k1":
            continue
        is_adaptive = arm.startswith("adaptive_k")
        ax = adaptive_ax if is_adaptive else fixed_ax
        k = int(arm.split("_k")[1])
        color = plt.colormaps["viridis"]((k - min_k) / (max_k - min_k))
        points = aggregate_concurrency(
            [row for row in rows if row["arm"] == arm],
            ("interactivity_tok_s_user", "output_throughput"),
        )
        plot_series(
            ax,
            points,
            "interactivity_tok_s_user",
            "output_throughput",
            marker="o",
            color=color,
            label=f"K={k}",
        )
        for row in points:
            ax.annotate(
                str(row["concurrency"]),
                (row["interactivity_tok_s_user"], row["output_throughput"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )
    fixed_ax.set_title("Fixed verification")
    adaptive_ax.set_title("Adaptive verification")
    fixed_ax.set_ylabel("Aggregate output throughput (tok/s)")
    for ax in (fixed_ax, adaptive_ax):
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.supxlabel("Interactivity (tok/s/user; 1000 / median TPOT in ms)")
    fig.suptitle(
        "Qwen3-32B-FP8 + EAGLE3, TP=2"
        " (point labels: concurrency; means of completed runs)"
    )
    save_figure(fig, HERE / "interactivity_throughput")


if __name__ == "__main__":
    main()
