"""Build the standalone GLM report from saved evaluations and metric deltas."""

import argparse
import hashlib
import html
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

from build_logs import build_logs

os.environ.setdefault("MPLCONFIGDIR", "/tmp/glm53-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from reporting.plots import is_pareto, plot_series, save_figure, write_csv

CODE = Path(__file__).resolve().parent
HERE = CODE.parent / "report"
SHA = "58b3298457dde7b4554b3b4e20b238c0ac2c3a65"
STYLES = {
    "baseline": ("No speculation", "#2563eb", "s"),
    "avoff": ("DSpark · AV off", "#d97706", "o"),
    "avon": ("DSpark · AV on", "#08916d", "D"),
}
TPOT = "vllm:request_time_per_output_token_seconds"


def main():
    global HERE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument(
        "--sources", type=Path, help="Optional JSON mapping modes to run directories"
    )
    args = parser.parse_args()
    HERE = args.output.resolve()
    HERE.mkdir(parents=True, exist_ok=True)
    if args.sources:
        import_results(json.loads(args.sources.read_text()))
    rows = load_rows()
    if not rows:
        raise RuntimeError("No completed evaluations found")
    for metric in ("mean", "p90"):
        for row in rows:
            row[f"pareto_{metric}"] = is_pareto(
                row, rows, "interactivity_" + metric, "throughput"
            )
    plot_frontier(rows, "mean")
    (HERE / "results.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n"
    )
    write_csv(rows, HERE / "results.csv")
    if (HERE / "logs").is_dir():
        build_logs(HERE)
    shutil.copy2(CODE / "README.md", HERE / "README.md")
    render_report(rows)
    manifest = {
        str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((HERE / "data").rglob("*"))
        if p.is_file()
    }
    (HERE / "data-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built {len(rows)} measured points: {HERE / 'index.html'}")
    for row in rows:
        print(
            f"{row['mode']:8} c{row['concurrency']:3}: {row['throughput']:8.1f} tok/s, "
            f"TPOT {row['mean_tpot_ms']:6.2f} ms, "
            f"interactivity {row['interactivity_mean']:6.1f}, "
            f"frontier={row['pareto_mean']}"
        )


def import_results(source_map):
    """Copy completed runs; the report remains rebuildable after /tmp is gone."""
    for mode, sources in source_map.items():
        for source in map(Path, sources):
            summary = source / "result.json"
            if not summary.exists():
                continue
            provenance = json.loads(summary.read_text())
            assert provenance["commit"] == SHA
            for run in provenance["runs"]:
                concurrency = run["concurrency"]
                destination = HERE / "data" / mode / f"c{concurrency}"
                destination.mkdir(parents=True, exist_ok=True)
                prefix = f"gsm8k-c{concurrency}"
                for suffix in (
                    ".json",
                    "-metrics-before.txt",
                    "-metrics-after.txt",
                    "-client.log",
                ):
                    shutil.copy2(
                        source / f"{prefix}{suffix}", destination / f"{prefix}{suffix}"
                    )
                shutil.copy2(source / "run.py", destination / "run.py")
                (destination / "provenance.json").write_text(
                    json.dumps(
                        {
                            k: v
                            for k, v in provenance.items()
                            if k not in ("runs", "status")
                        },
                        indent=2,
                    )
                    + "\n"
                )


def snapshot(path):
    sums = {}
    for line in path.read_text().splitlines():
        match = re.match(r"(vllm:[\w]+)(?:\{(.*?)\})?\s+([\d.eE+\-]+|\+Inf)$", line)
        if match:
            name, labels, value = match.groups()
            le = re.search(r'\ble="([^"]+)"', labels or "")
            key = (name, float(le.group(1)) if le else None)
            sums[key] = sums.get(key, 0.0) + float(value)
    return sums


def quantile(delta, metric, q):
    count = delta[(metric + "_count", None)]
    target = q * count
    previous_bound = previous_count = 0.0
    for bound, cumulative in sorted(
        (le, v) for (name, le), v in delta.items() if name == metric + "_bucket"
    ):
        if cumulative >= target:
            if not math.isfinite(bound):
                raise ValueError("Quantile lies in the unbounded final bucket")
            estimate = previous_bound + (bound - previous_bound) * (
                target - previous_count
            ) / (cumulative - previous_count)
            return estimate, previous_bound, bound
        previous_bound, previous_count = bound, cumulative
    raise ValueError("Missing histogram buckets")


def load_rows():
    rows = []
    for mode in STYLES:
        for path in sorted((HERE / "data" / mode).glob("c*/gsm8k-c*.json")):
            result = json.loads(path.read_text())
            prefix = path.with_suffix("")
            before = snapshot(Path(str(prefix) + "-metrics-before.txt"))
            after = snapshot(Path(str(prefix) + "-metrics-after.txt"))
            delta = {k: v - before.get(k, 0) for k, v in after.items()}
            count = delta[(TPOT + "_count", None)]
            assert count == result["num_questions"] == 1319, (path, count)
            assert all(
                v >= -1e-8
                for (name, _), v in delta.items()
                if name.endswith(("_total", "_sum", "_count", "_bucket"))
            ), path
            mean_tpot = delta[(TPOT + "_sum", None)] / count
            p90, lower, upper = quantile(delta, TPOT, 0.9)
            ttft = "vllm:time_to_first_token_seconds"
            e2e = "vllm:e2e_request_latency_seconds"
            for name in (ttft, e2e):
                assert delta[(name + "_count", None)] == count
            row = {
                "mode": mode,
                "label": STYLES[mode][0],
                "concurrency": result["concurrency"],
                "throughput": result["tokens_per_second"],
                "accuracy_pct": result["accuracy"] * 100,
                "invalid_count": round(result["invalid_rate"] * count),
                "questions": int(count),
                "duration_s": result["latency"],
                "output_tokens": result["total_output_tokens"],
                "mean_tpot_ms": mean_tpot * 1000,
                "interactivity_mean": 1 / mean_tpot,
                "p90_tpot_ms_estimate": p90 * 1000,
                "p90_tpot_lower_ms": lower * 1000,
                "p90_tpot_upper_ms": upper * 1000,
                "interactivity_p90": 1 / p90,
                "interactivity_p90_lower": 1 / upper,
                "interactivity_p90_upper": 1 / lower if lower else None,
                "mean_ttft_ms": delta[(ttft + "_sum", None)] / count * 1000,
                "mean_e2e_s": delta[(e2e + "_sum", None)] / count,
                "acceptance_length": result["mean_acceptance_length"],
                "draft_acceptance_pct": result["draft_acceptance_rate"] * 100
                if result["draft_acceptance_rate"] is not None
                else None,
                "source": str(path.relative_to(HERE)),
                "timestamp": result["timestamp"],
            }
            rows.append(row)
    return sorted(rows, key=lambda r: (r["concurrency"], list(STYLES).index(r["mode"])))


def figure():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#000000",
            "text.color": "#000000",
            "axes.titleweight": "bold",
        }
    )
    fig, ax = plt.subplots(figsize=(10.5, 6.3), layout="constrained")
    ax.grid(color="#dbeafe", alpha=1)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    return fig, ax


def save_plot(fig, name):
    save_figure(fig, HERE / "plots" / name)


def plot_frontier(rows, metric):
    fig, ax = figure()
    key = "interactivity_" + metric
    for mode, (label, color, marker) in STYLES.items():
        points = [r for r in rows if r["mode"] == mode]
        plot_series(
            ax,
            points,
            key,
            "throughput",
            color=color,
            linewidth=2,
            label=label,
        )
        for row in points:
            x, y = row[key], row["throughput"]
            if metric == "p90" and row["interactivity_p90_upper"] is not None:
                ax.errorbar(
                    x,
                    y,
                    xerr=[
                        [x - row["interactivity_p90_lower"]],
                        [row["interactivity_p90_upper"] - x],
                    ],
                    color=color,
                    alpha=0.25,
                    capsize=3,
                    linewidth=1,
                )
            ax.plot(
                x,
                y,
                marker=marker,
                color=color,
                markersize=7,
                linestyle="none",
                gid=f"point-{metric}-{mode}-{row['concurrency']}",
            )
            label_below = mode == "avoff"
            if row["concurrency"] == 64:
                label_below = not label_below
            ax.annotate(
                f"c{row['concurrency']}",
                (x, y),
                xytext=(6, -14 if label_below else 7),
                textcoords="offset points",
                fontsize=9,
                color="#000000",
            )
    tpot_label = (
        "mean request TPOT" if metric == "mean" else "estimated p90 request TPOT"
    )
    ax.set(
        xlabel=f"Interactivity: 1 / {tpot_label} (tok/s/user) →",
        ylabel="Aggregate output throughput (tok/s) ↑",
        title="Throughput versus interactivity",
    )
    xmax = max(
        max(r[key], r["interactivity_p90_upper"] or 0) if metric == "p90" else r[key]
        for r in rows
    )
    ax.set_xlim(0, xmax * 1.18)
    ax.set_ylim(0, max(r["throughput"] for r in rows) * 1.14)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="outside lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("GLM-5.3 NVFP4 · 4× B300 · DEP4 · GSM8K", fontsize=11, color="#000000")
    save_plot(fig, "pareto-" + metric)


def plot_concurrency(rows):
    fig, ax = figure()
    for mode, (label, color, marker) in STYLES.items():
        points = [r for r in rows if r["mode"] == mode]
        plot_series(
            ax,
            points,
            "concurrency",
            "throughput",
            marker=marker,
            color=color,
            label=label,
        )
    ax.set(
        xscale="log",
        xlabel="Total client concurrency",
        ylabel="Aggregate output throughput (tok/s)",
        title="Throughput across the measured concurrency sweep",
    )
    ax.set_xticks([8, 64, 128, 256], ["8", "64", "128", "256"])
    ax.set_ylim(bottom=0)
    ax.legend()
    save_plot(fig, "throughput-concurrency")


def svg(name, rows):
    text = (HERE / "plots" / f"{name}.svg").read_text()
    text = text[text.index("<svg") :]
    for row in rows:
        for metric in ("mean", "p90"):
            identifier = f"point-{metric}-{row['mode']}-{row['concurrency']}"
            title = (
                f"{row['label']} · concurrency {row['concurrency']} · "
                f"{row['throughput']:,.1f} tok/s · "
                f"{row['interactivity_' + metric]:.1f} tok/s/user · "
                f"accuracy {row['accuracy_pct']:.2f}%"
            )
            text = text.replace(
                f'<g id="{identifier}">',
                f'<g id="{identifier}"><title>{html.escape(title)}</title>',
            )
    return text


def render_report(rows):
    def number(value, digits=2):
        return "N/A" if value is None else f"{value:,.{digits}f}"

    labels = {
        "baseline": "No speculation",
        "avoff": "DSpark, AV off",
        "avon": "DSpark + AV",
    }
    throughput_table = []
    for concurrency in sorted({row["concurrency"] for row in rows}):
        points = {row["mode"]: row for row in rows if row["concurrency"] == concurrency}
        cells = [str(concurrency)] + [
            number(points[mode]["throughput"], 1) if mode in points else "N/A"
            for mode in labels
        ]
        throughput_table.append(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"
        )
    accuracy_table = []
    for row in sorted(
        rows, key=lambda r: (list(labels).index(r["mode"]), r["concurrency"])
    ):
        accuracy_table.append(
            "<tr>"
            + "".join(
                f"<td>{cell}</td>"
                for cell in (
                    html.escape(labels[row["mode"]]),
                    row["concurrency"],
                    number(row["accuracy_pct"]) + "%",
                    number(row["acceptance_length"], 3),
                )
            )
            + "</tr>"
        )
    replacements = {
        "@@THEME@@": (CODE.parents[2] / "reporting/b200-theme.css").read_text(),
        "@@THROUGHPUT_TABLE@@": "\n".join(throughput_table),
        "@@ACCURACY_TABLE@@": "\n".join(accuracy_table),
        "@@PARETO_MEAN@@": svg("pareto-mean", rows),
    }
    page = (CODE / "template.html").read_text()
    for old, new in replacements.items():
        page = page.replace(old, new)
    assert "@@" not in page
    (HERE / "index.html").write_text(page)


if __name__ == "__main__":
    main()
