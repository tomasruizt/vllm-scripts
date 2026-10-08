"""Add DeepSeek-V4-Flash results to the existing GLM DEP4 report."""

import argparse
import hashlib
import html
import json
import shutil
import sys
import tarfile
from pathlib import Path
from statistics import stdev

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE.parent / "glm53_av/analysis"))
import build_report as glm

from reporting.model_tabs import model_tabs
from reporting.plots import (
    aggregate_concurrency,
    plot_series,
    save_figure,
    scatter_series,
    write_csv,
)

STYLES = {mode: glm.STYLES[mode] for mode in ("avon", "reclaim")}


def import_inputs(sources, output):
    for method, source in sources.items():
        source = Path(source)
        result = json.loads((source / "result.json").read_text())
        assert result["status"] == "completed" and len(result["runs"]) == 60
        destination = output / "logs" / method
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("result.json", "server.log"):
            shutil.copy2(source / name, destination / name)
        for run in result["runs"]:
            summary = Path(run["aiperf_summary"])
            target = destination / summary
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / summary, target)
            for suffix in (
                "-client.log",
                "-command.json",
                "-cache-reset.json",
                "-metrics-before.txt",
                "-metrics-after.txt",
            ):
                filename = summary.parent.name + suffix
                shutil.copy2(source / filename, destination / filename)


def load_runs(output):
    rows = []
    for method in STYLES:
        folder = output / "logs" / method
        result = json.loads((folder / "result.json").read_text())
        assert result["status"] == "completed"
        assert result["budget_expansion_seen"] == (method == "reclaim")
        for run in result["runs"]:
            summary = Path(run["aiperf_summary"])
            report = json.loads((folder / summary).read_text())
            assert report["request_count"]["avg"] == 1536
            assert not (report.get("error_request_count") or {}).get("avg", 0)
            assert (
                report["output_sequence_length"]["min"]
                == report["output_sequence_length"]["max"]
                == 512
            )
            label = summary.parent.name
            before = glm.snapshot(folder / f"{label}-metrics-before.txt")
            after = glm.snapshot(folder / f"{label}-metrics-after.txt")
            count = (
                after[(glm.TPOT + "_count", None)] - before[(glm.TPOT + "_count", None)]
            )
            assert count == 1536
            assert (
                json.loads((folder / f"{label}-cache-reset.json").read_text())[
                    "success"
                ]
                is True
            )
            rows.append(
                {
                    "mode": method,
                    "concurrency": run["concurrency"],
                    "repeat": run["repeat"],
                    "throughput": report["output_token_throughput"]["avg"],
                    "mean_tpot_ms": 1000
                    * (
                        after[(glm.TPOT + "_sum", None)]
                        - before[(glm.TPOT + "_sum", None)]
                    )
                    / count,
                    "acceptance_length": run["mean_acceptance_length"],
                }
            )
    for row in rows:
        row["interactivity_mean"] = 1000 / row["mean_tpot_ms"]
    return rows


def aggregate(runs):
    rows = []
    for method, (label, _, _) in STYLES.items():
        points = [r for r in runs if r["mode"] == method]
        for row in aggregate_concurrency(
            points, ["throughput", "mean_tpot_ms", "acceptance_length"]
        ):
            repeats = [r for r in points if r["concurrency"] == row["concurrency"]]
            assert len(repeats) == 10
            row.update(
                mode=method,
                label=label,
                num_runs=len(repeats),
                interactivity_mean=1000 / row["mean_tpot_ms"],
            )
            for key in ("throughput", "mean_tpot_ms"):
                row[key + "_sd"] = stdev(r[key] for r in repeats)
            rows.append(row)
    return rows


def plot(rows, output, *, runs):
    fig, ax = glm.figure()
    for method, (label, color, marker) in STYLES.items():
        points = [r for r in rows if r["mode"] == method]
        scatter_series(
            ax,
            [r for r in runs if r["mode"] == method],
            "interactivity_mean",
            "throughput",
            color=color,
            marker=marker,
            s=24,
            alpha=0.7,
            gid=f"runs-mean-{method}",
        )
        plot_series(
            ax,
            points,
            "interactivity_mean",
            "throughput",
            label=label,
            color=color,
            linewidth=2,
        )
        for row in points:
            x, y = row["interactivity_mean"], row["throughput"]
            ax.plot(
                x,
                y,
                marker=marker,
                color=color,
                markersize=7,
                linestyle="none",
                gid=f"point-mean-{method}-{row['concurrency']}",
            )
            ax.annotate(
                f"c{row['concurrency']}",
                (x, y),
                xytext=(6, 7 if method == "avon" else -18),
                textcoords="offset points",
                fontsize=9,
                color="#000000",
            )
    ax.set(
        xlabel="Interactivity: 1 / mean request TPOT (tok/s/user) →",
        ylabel="Aggregate output throughput (tok/s) ↑",
        title="Throughput versus interactivity",
        xlim=(0, max(r["interactivity_mean"] for r in rows + runs) * 1.18),
        ylim=(0, max(r["throughput"] for r in rows + runs) * 1.14),
    )
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="outside lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    save_figure(fig, output / "plots/pareto-mean")


def export_logs(output):
    files = sorted(
        p
        for p in (output / "logs").rglob("*")
        if p.is_file() and p != output / "logs/sha256.json"
    )
    manifest = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    (output / "logs/sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with tarfile.open(output / "benchmark-logs.tar.gz", "w:gz") as archive:
        archive.add(output / "logs", arcname="logs")
    links = "".join(
        f'<li><a href="{html.escape(name)}">{html.escape(name)}</a></li>'
        for name in manifest
    )
    theme = (CODE.parents[1] / "reporting/b200-theme.css").read_text()
    (output / "benchmark-logs.html").write_text(
        f'<!doctype html><html><head><meta charset="utf-8"><style>{theme}</style></head><body><h1>DeepSeek benchmark and server logs</h1><p><a href="../index.html">Report</a> · <a href="benchmark-logs.tar.gz">Download all logs</a></p><ul>{links}</ul></body></html>'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glm-report", type=Path, required=True)
    parser.add_argument(
        "--sources",
        type=Path,
        help="Local JSON mapping avon/reclaim to completed run directories",
    )
    args = parser.parse_args()
    output = args.glm_report.resolve() / "deepseek-v4"
    output.mkdir(parents=True, exist_ok=True)
    if args.sources:
        import_inputs(json.loads(args.sources.read_text()), output)
    runs = load_runs(output)
    rows = aggregate(runs)
    (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
    write_csv(rows, output / "results.csv")
    export_logs(output)
    plot(rows, output, runs=runs)
    glm.HERE = args.glm_report.resolve()
    glm_rows = json.loads((glm.HERE / "results.json").read_text())
    glm_runs = json.loads((glm.HERE / "runs.json").read_text())
    glm.plot_frontier(glm_rows, "mean", runs=glm_runs)
    glm.render_report(glm_rows)
    glm_page = (glm.HERE / "index.html").read_text()
    glm.HERE = output
    url = "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/tree/62af8fffb2f7030cac4de2f0169f5b8d1101b646"
    glm.render_report(
        rows,
        metadata={
            "secondary_metric": "mean_tpot_ms",
            "replacements": {
                "@@TITLE@@": "DeepSeek-V4-Flash · DEP4 · adaptive verification",
                "@@HEADING@@": "DeepSeek-V4-Flash: throughput versus interactivity",
                "@@SUBTITLE@@": "4× B300 · TP1 / DP4 / EP4 · vLLM main 58b329845 · AIPerf · SPEED-Bench throughput_1k",
                "@@MODELS@@": f'Target: <a href="{url}">deepseek-ai/DeepSeek-V4-Flash-DSpark</a><br>Draft: native DSpark module bundled in the same checkpoint · 7 draft tokens',
                "@@NOTE@@": '<p class="metric-note">The first three baseline repeats were slower across most concurrency levels. Full averages include all ten repeats; the apparent gain may reflect that slowdown. These runs do not establish a reliable reclamation benefit.</p>',
                "@@CAPTION@@": "Small points show the ten individual runs at each concurrency (opacity 0.7); curves connect the means. Each run uses 1,536 SPEED-Bench prompts and exactly 512 output tokens per request. Prefix caching enabled and cleared before every run. Interactivity is 1 / mean request TPOT. Labels show total client concurrency. Methods ran in separate blocks; this is a performance workload, not an accuracy evaluation.",
                "@@SECONDARY_LABEL@@": "TPOT (ms/token)",
            },
        },
    )
    page = (output / "index.html").read_text()
    for name in ("benchmark-logs.html", "benchmark-logs.tar.gz"):
        page = page.replace(f'href="{name}"', f'href="deepseek-v4/{name}"')
    combined = model_tabs(
        [("glm", "GLM-5.3", glm_page), ("deepseek", "DeepSeek-V4-Flash", page)],
        title="DEP4 · adaptive verification and padding reclamation",
    )
    (args.glm_report / "index.html").write_text(combined)
    print(args.glm_report / "index.html")


if __name__ == "__main__":
    main()
