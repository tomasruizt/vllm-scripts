"""Build an offline B300 AV report from the completed concurrency sweep."""

import argparse
import html
import json
import os
import shutil
import statistics
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nemotron-report-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prometheus_client.parser import text_string_to_metric_families

CODE = Path(__file__).resolve().parent
REPO = CODE.parents[2]
sys.path.insert(0, str(REPO))
from reporting.plots import plot_series, save_figure, write_csv

TPOT = "vllm:request_time_per_output_token_seconds"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Completed sweep results.json")
    parser.add_argument(
        "--comparison",
        type=Path,
        action="append",
        default=[],
        help="Additional completed sweep results.json",
    )
    parser.add_argument("--output", type=Path, default=CODE.parent / "report")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archived = output / "results.json"
    data = load_sweep(args.source, archived)
    comparisons = output / "data/comparisons"
    for source in args.comparison:
        k = speculative_config(json.loads(source.read_text()))["num_speculative_tokens"]
        load_sweep(source, comparisons / f"k{k}" / "results.json")
    datasets = [data] + [
        load_sweep(None, path) for path in sorted(comparisons.glob("*/results.json"))
    ]
    datasets.sort(key=lambda d: speculative_config(d)["num_speculative_tokens"])
    assert len(
        {speculative_config(d)["num_speculative_tokens"] for d in datasets}
    ) == len(datasets)
    for other in datasets:
        for key in (
            "model",
            "commit",
            "cases",
            "n_runs",
            "max_tokens",
            "temperature",
            "seed",
        ):
            assert other[key] == data[key], key
        assert speculative_config(other)["method"] == speculative_config(data)["method"]
        assert speculative_config(other).get("model") == speculative_config(data).get(
            "model"
        )
    rows = [row for dataset in datasets for row in summarize(dataset)]
    write_csv(rows, output / "summary.csv")
    for k in sorted({r["k"] for r in rows}):
        draw(rows, output, k)
    render(datasets, rows, output)
    print(output / "index.html")


def summarize(data):
    assert data["status"] == "complete"
    assert len(data["runs"]) == len(data["cases"]) * 2 * data["n_runs"]
    rows = []
    for concurrency, questions in sorted((int(c), n) for c, n in data["cases"].items()):
        row = {
            "k": speculative_config(data)["num_speculative_tokens"],
            "concurrency": concurrency,
            "questions": questions,
        }
        for mode in ("no-av", "av"):
            runs = [
                r
                for r in data["runs"]
                if r["mode"] == mode and r["concurrency"] == concurrency
            ]
            assert sorted(r["repeat"] for r in runs) == list(
                range(1, data["n_runs"] + 1)
            )
            assert all(r["questions"] == questions for r in runs)
            row[mode] = statistics.mean(r["tokens_per_second"] for r in runs)
            row[mode + "_sd"] = statistics.stdev(r["tokens_per_second"] for r in runs)
            row[mode + "_al"] = statistics.mean(
                r["mean_acceptance_length"] for r in runs
            )
            row[mode + "_interactivity"] = sum(r["tpot_count"] for r in runs) / sum(
                r["tpot_sum"] for r in runs
            )
        row["change_percent"] = 100 * (row["av"] / row["no-av"] - 1)
        rows.append(row)
    return rows


def draw(rows, output, k):
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"}
    )
    fig, ax = plt.subplots(figsize=(10.5, 4.5), layout="constrained")
    points = [r for r in rows if r["k"] == k]
    for mode, label, color, marker in (
        ("no-av", "AV off", "#d97706", "o"),
        ("av", "AV on", "#08916d", "D"),
    ):
        plot_series(
            ax,
            points,
            mode + "_interactivity",
            mode,
            label=label,
            color=color,
            marker=marker,
            linewidth=2,
        )
        ax.errorbar(
            [r[mode + "_interactivity"] for r in points],
            [r[mode] for r in points],
            yerr=[r[mode + "_sd"] for r in points],
            fmt="none",
            ecolor=color,
            capsize=4,
        )
        for row in points:
            ax.annotate(
                f"c{row['concurrency']}",
                (row[mode + "_interactivity"], row[mode]),
                xytext=(-6, -14) if mode == "no-av" else (6, 8),
                ha="right" if mode == "no-av" else "left",
                textcoords="offset points",
                fontsize=9,
            )
    ax.set(
        xlabel="Interactivity: 1 / mean request TPOT (tok/s/user)",
        ylabel="Aggregate output throughput (tok/s)",
        xlim=(
            0,
            max(r[m + "_interactivity"] for r in rows for m in ("no-av", "av")) * 1.15,
        ),
        ylim=(0, max(r[m] for r in rows for m in ("no-av", "av")) * 1.12),
    )
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    save_figure(fig, output / f"throughput-interactivity-k{k}")


def render(datasets, rows, output):
    data = datasets[0]
    spec = speculative_config(data)
    method = {"dspark": "DSpark", "mtp": "MTP"}[spec["method"]]
    draft = (
        model_link(spec["model"])
        + "; W4A16_NVFP4 draft weights, with selected modules unquantized. AV uses the online acceptance estimator; this checkpoint has no confidence head."
        if spec["method"] == "dspark"
        else "The target checkpoint’s built-in MTP module; no separate draft checkpoint."
    )
    sections = []
    lengths = sorted({r["k"] for r in rows})
    for k in lengths:
        svg = (output / f"throughput-interactivity-k{k}.svg").read_text()
        chart = svg[svg.index("<svg") :]
        table = []
        for r in (r for r in rows if r["k"] == k):
            table.append(
                f'<tr><th scope="row">{r["concurrency"]}</th><td>{r["questions"]:,}</td>'
                f"<td>{r['no-av']:,.0f}</td><td>{r['av']:,.0f}</td>"
                f"<td>{r['change_percent']:+.1f}%</td><td>{r['av_al']:.2f} / {r['no-av_al']:.2f}</td></tr>"
            )
        sections.append(
            f'<section class="tab-panel" id="panel-k{k}"><section class="card">'
            f'<figure aria-label="K={k}: throughput versus interactivity">{chart}'
            f"<figcaption>Interactivity is 1 / mean request TPOT, pooling Prometheus sum/count deltas across {data['n_runs']} runs. "
            "Throughput is the mean across runs; vertical error bars show sample standard deviation. "
            "Labels show client concurrency; both plots use the same axis limits.</figcaption></figure></section>"
            f'<section class="card"><h2>K={k} results</h2><div class="table-scroll"><table>'
            '<thead><tr><th scope="col">Concurrency</th><th scope="col">Questions / run</th>'
            '<th scope="col">AV off (tok/s)</th><th scope="col">AV on (tok/s)</th>'
            '<th scope="col">AV change</th><th scope="col">AL: on / off</th></tr></thead>'
            f"<tbody>{''.join(table)}</tbody></table></div>"
            f'<p class="metric-note">Throughput: mean across {data["n_runs"]} runs. AL: mean of per-run acceptance lengths.</p></section></section>'
        )
    choices = "".join(
        f'<input class="metric-choice" type="radio" name="draft-length" id="draft-k{k}" '
        f'aria-controls="panel-k{k}" {"checked" if k == lengths[0] else ""}>'
        for k in lengths
    )
    labels = "".join(f'<label for="draft-k{k}">K={k}</label>' for k in lengths)
    tab_css = "\n".join(
        f'#draft-k{k}:checked~.metric-tabs label[for="draft-k{k}"]{{background:#235cca;color:white}}\n'
        f"#draft-k{k}:checked~.panels>#panel-k{k}{{display:block}}\n"
        f'#draft-k{k}:focus-visible~.metric-tabs label[for="draft-k{k}"]{{outline:3px solid #e4ad31;outline-offset:3px}}'
        for k in lengths
    )
    values = {
        "THEME": (REPO / "reporting/b200-theme.css").read_text(),
        "TAB_CSS": tab_css,
        "DRAFT_SELECTOR": f'<div class="draft-selector">{choices}<div class="metric-tabs model-tabs">{labels}</div><div class="panels">{"".join(sections)}</div></div>',
        "PLOT_LINKS": " · ".join(
            f'<a href="throughput-interactivity-k{k}.svg">K{k} SVG</a> · <a href="throughput-interactivity-k{k}.png">K{k} PNG</a>'
            for k in lengths
        ),
        "COMMIT": html.escape(data["commit"][:9]),
        "DATE": html.escape(data["started"][:10]),
        "METHOD": method,
        "DRAFT": draft,
        "TARGET": model_link(data["model"]),
        "K": " / ".join(str(k) for k in lengths),
        "DATA_LINKS": " · ".join(
            f'<a href="{path.relative_to(output)}">K{speculative_config(json.loads(path.read_text()))["num_speculative_tokens"]} raw runs and commands</a>'
            for path in [
                output / "results.json",
                *sorted((output / "data/comparisons").glob("*/results.json")),
            ]
        ),
        "BLOCK_NOTE": (
            "<p>This checkpoint advertises an eight-slot draft block. K=15 evaluates a longer block (15 draft tokens plus the anchor) with the same weights.</p>"
            if spec["method"] == "dspark" and 15 in lengths
            else ""
        ),
        "RUNS": str(data["n_runs"]),
    }
    document = (CODE / "template.html").read_text()
    for key, value in values.items():
        document = document.replace("@@" + key + "@@", value)
    assert "@@" not in document
    (output / "index.html").write_text(document)


def load_sweep(source, archived):
    archived.parent.mkdir(parents=True, exist_ok=True)
    if source and source.resolve() != archived.resolve():
        shutil.copy2(source, archived)
    data = json.loads(archived.read_text())
    metrics_dir = archived.parent / "data"
    metrics_dir.mkdir(exist_ok=True)
    for run in data["runs"]:
        prefix = f"{run['mode']}-c{run['concurrency']}-run{run['repeat']}"
        for side in ("before", "after"):
            name = prefix + f"-{side}.txt"
            destination = metrics_dir / name
            if source:
                snapshot = source.parent / name
                if not snapshot.exists():
                    snapshot = source.parent / "data" / name
                if snapshot.resolve() != destination.resolve():
                    shutil.copy2(snapshot, destination)
        before = tpot_metrics(metrics_dir / (prefix + "-before.txt"))
        after = tpot_metrics(metrics_dir / (prefix + "-after.txt"))
        run["tpot_count"] = after["count"] - before["count"]
        run["tpot_sum"] = after["sum"] - before["sum"]
        assert run["tpot_count"] == run["questions"], prefix
        assert run["tpot_sum"] > 0, prefix
    return data


def speculative_config(data):
    command = data["no-av_command"]
    return json.loads(command[command.index("--speculative-config") + 1])


def model_link(path):
    cached = next(
        (part for part in Path(path).parts if part.startswith("models--")), None
    )
    repo = cached.removeprefix("models--").replace("--", "/") if cached else path
    return f'<a href="https://huggingface.co/{html.escape(repo, quote=True)}">{html.escape(repo)}</a>'


def tpot_metrics(path):
    values = {"sum": 0.0, "count": 0.0}
    for family in text_string_to_metric_families(path.read_text()):
        for sample in family.samples:
            for suffix in values:
                if sample.name == TPOT + "_" + suffix:
                    values[suffix] += sample.value
    return values


if __name__ == "__main__":
    main()
