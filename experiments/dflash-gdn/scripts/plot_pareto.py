"""Plot measured tradeoff curves and Pareto-optimal points for either model."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

SERIES = {
    "vllm_baseline": ("vLLM baseline", "#2563eb", "^", "--"),
    "vllm_dflash": ("vLLM + DFlash", "#2563eb", "o", "-"),
    "sglang_baseline": ("SGLang baseline", "#08916d", "v", "--"),
    "sglang_dflash": ("SGLang + DFlash", "#08916d", "s", "-"),
    "pr2_dflash": ("vLLM + DFlash PR 52297", "#d97706", "D", "-"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--model", choices=("4B", "27B", "35B-A3B"), default="4B")
    args = parser.parse_args()
    manifest = args.root / "experiment.json"
    proposals = json.loads(manifest.read_text())["num_speculative_tokens"] if manifest.exists() else 15
    rows = read_points(args.root, args.model)
    frontier = pareto_points(rows)
    out = args.root / "plots"
    out.mkdir(exist_ok=True)
    stem = out / f"{args.model.lower()}-throughput-interactivity"

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.20, top=0.84)
    fig.suptitle("Throughput vs interactivity", x=0.12, y=0.96, ha="left", fontsize=19, weight="bold")
    model_name = "Qwen3.6-35B-A3B" if args.model == "35B-A3B" else f"Qwen3.5-{args.model}"
    fig.text(0.12, 0.90, f"{model_name} · 1× B200 · FP8 · BF16 Mamba states · GSM8K · DFlash block {proposals + 1}", color="#263448")

    for variant, (label, color, marker, linestyle) in SERIES.items():
        points = [r for r in rows if r["variant"] == variant]
        ax.plot(
            [r["interactivity_tokens_s"] for r in points],
            [r["throughput_tokens_s"] for r in points],
            color=color, marker=marker, linestyle=linestyle,
            linewidth=2.2, markersize=6.5, label=label,
        )
        for point in points:
            if variant.endswith("baseline") and point["concurrency"] in (2, 4):
                continue
            offset = (7, 8) if variant != "pr2_dflash" else (7, -16)
            if variant == "vllm_baseline":
                offset = (-34, 2)
            elif variant == "sglang_baseline":
                offset = (8, 4)
            ax.annotate(
                f"c={point['concurrency']}",
                (point["interactivity_tokens_s"], point["throughput_tokens_s"]),
                xytext=offset, textcoords="offset points", fontsize=9, color=color,
            )
    ax.set_xlabel("Interactivity from TPOT p99 (tok/s/user)  → better", labelpad=12)
    ax.set_ylabel("Aggregate throughput (tok/s)  ↑ better", labelpad=12)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_ylim(bottom=0)
    ax.margins(x=0.10, y=0.10)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.text(0.12, 0.075, "Markers: c=1, 2, 4, 8, 16, 32. Dashed curves show the baselines.", fontsize=9, color="#263448")
    fig.text(0.12, 0.04, "Interactivity = 1,000 / TPOT p99 (ms). Single runs; connecting lines are visual guides.", fontsize=9, color="#263448")
    for extension in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=200, facecolor="white")
    plt.close(fig)
    with stem.with_suffix(".csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=[*rows[0], "pareto_optimal"])
        writer.writeheader()
        writer.writerows(dict(r, pareto_optimal=r in frontier) for r in rows)
    print(stem.with_suffix(".png"))
    print("Pareto points:", [(r["configuration"], r["concurrency"]) for r in frontier])


def read_points(root, model):
    rows = []
    for variant, (label, _, _, _) in SERIES.items():
        engine_mode = "vllm_dflash" if variant == "pr2_dflash" else variant
        for concurrency in (1, 2, 4, 8, 16, 32):
            path = root / model / variant / engine_mode / f"c{concurrency}" / "aiperf/profile_export_aiperf.json"
            report = json.loads(path.read_text())
            assert report["inter_token_latency"]["unit"] == "ms"
            tpot = report["inter_token_latency"]["p99"]
            rows.append({
                "variant": variant,
                "configuration": label,
                "concurrency": concurrency,
                "latency_p50_ms": report["request_latency"]["p50"],
                "throughput_tokens_s": report["output_token_throughput"]["avg"],
                "tpot_p99_ms": tpot,
                "interactivity_tokens_s": 1000 / tpot,
            })
    return rows


def pareto_points(rows):
    def dominates(a, b):
        ax, bx = a["interactivity_tokens_s"], b["interactivity_tokens_s"]
        ay, by = a["throughput_tokens_s"], b["throughput_tokens_s"]
        return ax >= bx and ay >= by and (ax > bx or ay > by)

    return [r for r in rows if not any(dominates(other, r) for other in rows)]


if __name__ == "__main__":
    main()
