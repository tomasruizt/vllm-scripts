#!/usr/bin/env python3
"""Summarize a vllm bench sweep serve experiment and plot its tradeoff."""

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "qwen3_32b_fp8_tp2_complete"


def main() -> None:
    rows = load_rows()
    if not rows:
        raise SystemExit(f"No completed benchmark runs under {RESULTS}")
    write_csv(rows)
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


def write_csv(rows: list[dict]) -> None:
    with (HERE / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
        by_concurrency = defaultdict(list)
        for row in rows:
            if row["arm"] == arm:
                by_concurrency[row["concurrency"]].append(row)
        points = sorted(
            (
                {
                    "concurrency": concurrency,
                    "interactivity_tok_s_user": mean(
                        row["interactivity_tok_s_user"] for row in runs
                    ),
                    "output_throughput": mean(
                        row["output_throughput"] for row in runs
                    ),
                }
                for concurrency, runs in by_concurrency.items()
            ),
            key=lambda row: row["concurrency"],
        )
        ax.plot(
            [row["interactivity_tok_s_user"] for row in points],
            [row["output_throughput"] for row in points],
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
    fig.savefig(HERE / "interactivity_throughput.png", dpi=180)
    fig.savefig(HERE / "interactivity_throughput.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
