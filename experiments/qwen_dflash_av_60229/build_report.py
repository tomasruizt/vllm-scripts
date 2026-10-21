"""Report AIPerf SPEED-Bench cells, measured in the steady-state window with validity gates."""

import argparse
import csv
from hashlib import sha256
import html
import json
import os
from pathlib import Path
import shlex
import sys
from statistics import mean, median

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qwen-av-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
MODE_SUFFIX = ""
sys.path.insert(0, str(REPO))
from reporting.plots import aggregate_concurrency, plot_series, scatter_series, save_figure, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="single run dir (one-tab report)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-csv", type=Path, help="Plot saved run-level mode/concurrency/interactivity/throughput rows")
    parser.add_argument("--series", action="append", help="mode:label:color:marker, repeatable for saved-run plots")
    parser.add_argument("--title", help="Title for saved-run plots")
    parser.add_argument("--figsize", nargs=2, type=float, default=(10.5, 4.5), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--mode-suffix", default="", help="run folders are <mode><suffix>, e.g. offcg/oncg with cg")
    parser.add_argument("--run", action="append", default=[],
                        help="slug:label:source[:suffix], repeatable, one tab per run (e.g. default:Default capture sizes:/runs/x)")
    args = parser.parse_args()
    if args.plot_csv:
        with args.plot_csv.open() as stream:
            rows = [dict(r, concurrency=int(r["concurrency"]), interactivity=float(r["interactivity"]),
                         throughput=float(r["throughput"]), cell="main", valid=True) for r in csv.DictReader(stream)]
        series = [tuple(spec.split(":")) for spec in args.series] if args.series else None
        draw(rows, args.output, series=series, title=args.title, figsize=tuple(args.figsize))
        return
    runs = [parse_run(spec) for spec in args.run] or [("run", "Results", args.source, args.mode_suffix)]
    args.output.mkdir(parents=True, exist_ok=True)
    panels = [build_run(slug, label, source, suffix, args.output) for slug, label, source, suffix in runs]
    render(args.output, panels)
    print(args.output / "index.html")


def parse_run(spec: str) -> tuple[str, str, Path, str]:
    slug, label, source, *suffix = spec.split(":")
    return slug, label, Path(source), suffix[0] if suffix else ""


def build_run(slug: str, label: str, source: Path, suffix: str, output: Path) -> dict:
    """Load one run, write its plot and CSVs to output/slug, and return its tab content."""
    global MODE_SUFFIX
    MODE_SUFFIX = suffix
    paths = sorted(source.glob("*/artifacts/**/profile_export_aiperf.json"))
    paths += sorted(source.glob("*/repeat-*/artifacts*/**/profile_export_aiperf.json"))
    modes = {path.relative_to(source).parts[0].removesuffix(MODE_SUFFIX) for path in paths}
    assert paths and "on" in modes, paths
    statuses = sorted(source.glob("worker-gpu*-exit-code.txt"))
    assert statuses and all(s.read_text().strip() == "0" for s in statuses), statuses
    records = [load_run(path, source) for path in paths]
    rows = sorted((summary(row) for row in records), key=lambda row: (row["concurrency"], row["mode"], row["repeat"]))
    assert len({(r["mode"], r["concurrency"]) for r in rows}) == len({r["mode"] for r in rows}) * len(levels(rows))
    (source / "quick-result.json").write_text(json.dumps(records) + "\n")
    out = output / slug
    out.mkdir(parents=True, exist_ok=True)
    draw(rows, out)
    write_csv(rows, out / "summary.csv")
    medians = aggregate(rows)
    write_csv(medians, out / "concurrency-summary.csv")
    (out / "summary.json").write_text(json.dumps(medians, indent=2) + "\n")
    return run_panel(slug, label, source, out, rows)


def load_run(path: Path, root: Path) -> dict:
    data = json.loads(path.read_text())
    metrics_path = path.with_name("server_metrics_export.json")
    server = json.loads(metrics_path.read_text())
    metrics = server["metrics"]
    request_path = path.with_name("profile_export.jsonl")
    requests = [json.loads(line) for line in request_path.read_text().splitlines()]
    done = [r for r in requests if "output_sequence_length" in r.get("metrics", {})]
    lengths = [r["metrics"]["output_sequence_length"]["value"] for r in done]
    phase = next(p for p in data["input_config"]["phases"] if p["name"] == "profiling")
    folder, repeat_dir, cell_dir = path.relative_to(root).parts[:3]
    mode = folder.removesuffix(MODE_SUFFIX)
    repeat, cell = int(repeat_dir.removeprefix("repeat-")), "sentinel" if cell_dir.endswith("-end") else "main"
    steady = steady_window(done, phase["concurrency"])
    row = dict(run_id=f"{root.name}/{mode}/{repeat_dir}/{cell_dir}",
               suite=root.name, phase="measurement", mode=mode, cell=cell,
               av_enabled=mode == "on", concurrency=phase["concurrency"], repeat=repeat,
               gpu=int((root / folder / repeat_dir / "gpu.txt").read_text()),
               requests_target=phase["requests"], requests=len(done), error_rate=data["request_error_rate"]["avg"],
               cancelled=sum(r["metadata"].get("was_cancelled", False) for r in requests),
               duration_s=data["benchmark_duration"]["avg"], aiperf_throughput=data["output_token_throughput"]["avg"],
               acceptance_length=1 + server_total(metrics, "vllm:spec_decode_num_accepted_tokens") / server_total(metrics, "vllm:spec_decode_num_drafts"),
               cap_hit_pct=100 * sum(n >= 1024 for n in lengths) / len(lengths),
               mean_output_tokens=data["output_sequence_length"]["avg"], mean_input_tokens=data["input_sequence_length"]["avg"],
               server_running_p50=server_stat(metrics, "vllm:num_requests_running", "p50"),
               server_waiting_p50=server_stat(metrics, "vllm:num_requests_waiting", "p50"),
               preemptions=server_total(metrics, "vllm:num_preemptions"),
               prompt_set=sha256("\n".join(sorted(r["metadata"]["conversation_id"] for r in requests)).encode()).hexdigest()[:12],
               source_json=str(path.relative_to(root)), dataset="nvidia/SPEED-Bench/throughput_1k",
               benchmark_client="AIPerf 0.13.0", natural_eos=True, output_len=1024,
               prefix_caching=False, temperature=0.0, seed=42,
               commit=(root / "commit.txt").read_text().strip(),
               instrumentation="none, unchanged PR60229", server_session=f"{mode}/{repeat_dir}",
               server_command=shlex.split((root / folder / repeat_dir / "server-command.txt").read_text()),
               aiperf_summary_json=json.dumps(data), aiperf_server_metrics_json=json.dumps(server),
               aiperf_request_records_json=json.dumps(requests), **steady)
    row.update(gates(row))
    outputs = path.with_name("outputs.json")
    row["aiperf_outputs_json"] = outputs.read_text() if outputs.exists() else None
    return row


def steady_window(requests: list[dict], concurrency: int) -> dict:
    """Metrics while the client holds `concurrency` requests in flight.

    The window opens at the `concurrency`-th completion (initial prefill burst is over) and closes
    when the last request is issued (drain begins). Throughput counts output tokens arriving inside it."""
    ends = sorted(r["metadata"]["request_end_ns"] for r in requests)
    start = ends[min(concurrency, len(ends)) - 1]
    end = max(r["metadata"]["request_start_ns"] for r in requests)
    if end <= start:
        return dict(window_s=0.0, window_requests=0, throughput=None, mean_tpot_ms=None, mean_itl_ms=None, mean_ttft_ms=None)
    tokens = sum(tokens_in_window(r, start, end) for r in requests)
    inside = [r["metrics"] for r in requests if start <= r["metadata"]["request_start_ns"] <= end]
    tpots = [m["inter_token_latency"]["value"] for m in inside if "inter_token_latency" in m]
    itls = [v for m in inside for v in m.get("inter_chunk_latency", {}).get("value", [])]
    return dict(window_s=(end - start) / 1e9, window_requests=len(inside), throughput=float(tokens) * 1e9 / (end - start),
                mean_tpot_ms=mean(tpots) if tpots else None,
                mean_itl_ms=mean(itls) if itls else None,
                mean_ttft_ms=mean(m["time_to_first_token"]["value"] for m in inside) if inside else None)


def tokens_in_window(record: dict, start: int, end: int) -> float:
    """Output tokens of one request arriving in [start, end), spread evenly over its streamed chunks."""
    metrics = record["metrics"]
    first = record["metadata"]["request_start_ns"] + metrics["time_to_first_token"]["value"] * 1e6
    gaps = metrics.get("inter_chunk_latency", {}).get("value", [])
    times = first + 1e6 * np.cumsum([0.0, *gaps])
    per_chunk = metrics["output_sequence_length"]["value"] / len(times)
    return per_chunk * np.count_nonzero((times >= start) & (times < end))


def gates(row: dict) -> dict:
    """A cell is valid only if it measured a saturated, unqueued, fully completed workload."""
    c = row["concurrency"]
    checks = dict(all_completed=row["requests"] == row["requests_target"] and row["cancelled"] == 0,
                  no_errors=row["error_rate"] == 0, no_preemptions=row["preemptions"] == 0,
                  server_saturated=row["server_running_p50"] >= 0.9 * c,
                  no_queueing=row["server_waiting_p50"] == 0, window_ge_5s=row["window_s"] >= 5)
    failed = [name for name, ok in checks.items() if not ok]
    return dict(valid=not failed, failed_gates=",".join(failed))


def server_total(metrics: dict, name: str) -> float:
    return sum(series["stats"]["total"] for series in metrics[name]["series"])


def server_stat(metrics: dict, name: str, stat: str) -> float:
    return max(series["stats"][stat] for series in metrics[name]["series"])


def levels(rows: list[dict]) -> list[int]:
    return sorted({r["concurrency"] for r in rows})


def summary(row: dict) -> dict:
    keys = ["mode", "cell", "concurrency", "repeat", "gpu", "valid", "failed_gates", "requests", "cancelled", "error_rate",
            "duration_s", "window_s", "window_requests", "throughput", "aiperf_throughput", "mean_tpot_ms", "mean_ttft_ms",
            "mean_itl_ms", "acceptance_length", "server_running_p50", "server_waiting_p50", "preemptions",
            "mean_input_tokens", "mean_output_tokens", "cap_hit_pct", "prompt_set"]
    result = {key: row[key] for key in keys}
    result["interactivity"] = 1000 / row["mean_tpot_ms"] if row["mean_tpot_ms"] else None
    return result


def measured(rows: list[dict]) -> list[dict]:
    """Main cells that passed every validity gate. Only these enter medians and the plot."""
    return [r for r in rows if r["cell"] == "main" and r["valid"]]


def aggregate(rows: list[dict]) -> list[dict]:
    return [
        {"concurrency": concurrency, "mode": mode, "runs": len(points),
         "cells": sum(r["cell"] == "main" and r["concurrency"] == concurrency and r["mode"] == mode for r in rows),
         **{key: median(row[key] for row in points) for key in
            ("requests", "throughput", "interactivity", "mean_tpot_ms", "mean_ttft_ms", "mean_itl_ms", "acceptance_length")}}
        for concurrency in levels(rows) for mode in ("off", "on")
        if (points := [r for r in measured(rows) if r["concurrency"] == concurrency and r["mode"] == mode])
    ]


def draw(rows: list[dict], output: Path, *, series: list[tuple[str, str, str, str]] | None = None,
         title: str | None = None, figsize: tuple[float, float] = (10.5, 4.5)) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    valid = measured(rows)
    medians = aggregate(rows) if series is None else []
    styles = series or [("off", "Median · AV off · k=7", "#d97706", "o"), ("on", "Median · AV on · k=7", "#08916d", "D")]
    for mode, label, color, marker in styles:
        runs = [r for r in valid if r["mode"] == mode]
        scatter_series(ax, runs, "interactivity", "throughput", color=color, marker=marker, alpha=0.4, s=42)
        grouped = aggregate_concurrency(runs, ["interactivity", "throughput"]) if series else [r for r in medians if r["mode"] == mode]
        plot_series(ax, grouped, "interactivity", "throughput", color=color, marker=marker, linewidth=2, label=label)
    for row in medians:
        ax.annotate(f"c{row['concurrency']}", (row["interactivity"], row["throughput"]), xytext=(7, 8), textcoords="offset points", fontsize=9)
    ax.set(xlabel="Interactivity: 1 / mean request TPOT (tok/s/user)", ylabel="Steady-state output throughput (tok/s)", xlim=(0, max(r["interactivity"] for r in valid) * 1.16), ylim=(0, max(r["throughput"] for r in valid) * 1.18))
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    if title:
        ax.set_title(title)
    save_figure(fig, output / "throughput-interactivity")


METRICS = [("throughput", "Output tok/s"), ("interactivity", "Tok/s/user"), ("mean_tpot_ms", "TPOT ms"),
           ("mean_ttft_ms", "TTFT mean ms"), ("mean_itl_ms", "ITL mean ms"), ("acceptance_length", "AL"),
           ("requests", "Requests / run")]


def render(output: Path, panels: list[dict]) -> None:
    """One page, one top-level tab per run (no tab bar when there is a single run)."""
    theme = (REPO / "reporting/b200-theme.css").read_text()
    rules = [rule for p in panels for rule in p["rules"]] + [
        rule for p in panels for rule in tab_rules(f"run-{p['slug']}", "run-tabs", f".run-panels>#panel-run-{p['slug']}")]
    choices = "".join(f'<input class="metric-choice" type="radio" name="run" id="run-{p["slug"]}" {"checked" if i == 0 else ""}>'
                      for i, p in enumerate(panels))
    labels = "".join(f'<label for="run-{p["slug"]}">{html.escape(p["label"])}</label>' for p in panels)
    bodies = "".join(f'<div class="tab-panel run-panel" id="panel-run-{p["slug"]}">{p["html"]}</div>' for p in panels)
    tabs = f'{choices}<div class="metric-tabs run-tabs">{labels}</div><div class="run-panels">{bodies}</div>' if len(panels) > 1 else panels[0]["html"]
    first = panels[0]["source"]
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Qwen3.5 + Adaptive Verification</title><style>{theme}
pre {{ overflow-x: auto; font-size: 12px; padding: 16px; background: #eef2f7; }}
.run-tabs {{ margin-bottom: 8px; }}
.run-panel {{ display: none; }}
{chr(10).join(rules)}
.comparison-table th.av-off,.comparison-table td.av-off {{ background: #fff4e5; font-weight: 650; }}
.comparison-table th.av-on,.comparison-table td.av-on {{ background: #e7f6f0; font-weight: 650; }}
.comparison-table th.delta,.comparison-table td.delta {{ background: #e9efff; color: #1746a2; font-weight: 750; }}
</style></head><body><h1>Qwen3.5 + Adaptive Verification</h1>
<p class="subtitle">35B-A3B BF16 · one B300 per server, TP=1 · DFlash k=7 · SPEED-Bench throughput-1K · AIPerf 0.13.0</p>
<p>Each cell sends a fixed number of requests at fixed concurrency and waits for all of them to finish (trailing c1 is a drift sentinel). Two workers on two GPUs alternate AV off and on per repeat with a fresh server each time, so both modes run on both GPUs. Throughput counts output tokens that arrive in the steady-state window: from the c-th completion until the last request is issued. TPOT, ITL and TTFT are means over requests that start inside that window (ITL over all their chunk gaps). Commit <code>{records_commit(first)}</code>.</p>
{tabs}
<footer><p>Native AIPerf SPEED-Bench loader, unchanged resolved throughput-1K dataset, seeded shuffle (42), so every cell uses a prefix of the same prompt order and AV off and on see identical prompts. Natural EOS, maximum 1,024 output tokens, temperature 0, thinking disabled. Prefix caching disabled. Gates: all requests completed and none cancelled, no errors, no preemptions, server running p50 ≥ 0.9·c, server waiting p50 = 0, window ≥ 5 s. AL uses accepted-token and draft-step totals from AIPerf's server-metric exports.</p></footer>
<details><summary>Benchmark config</summary><pre>{html.escape(public((first / "steady-speedbench.yaml").read_text()))}</pre><pre>aiperf profile --config steady-speedbench.yaml  # __C__, __N__ from the cell list</pre></details></body></html>'''
    (output / "index.html").write_text(page)


def run_panel(slug: str, label: str, source: Path, out: Path, rows: list[dict]) -> dict:
    """Tab content for one run: capture setting, plot, metric tables and server commands. Links point into slug/."""
    svg = (out / "throughput-interactivity.svg").read_text()
    svg = svg[svg.index("<svg"):]
    medians = aggregate(rows)
    choices, labels, panels, rules = [], [], [], []
    for index, (key, name) in enumerate(METRICS):
        tab = f"{slug}-metric-{key}"
        choices.append(f'<input class="metric-choice" type="radio" name="{slug}-table" id="{tab}" {"checked" if index == 0 else ""}>')
        labels.append(f'<label for="{tab}">{name}</label>')
        rules.extend(tab_rules(tab, "metric-tabs", f".panels>#panel-{tab}"))
        body = [comparison_row(concurrency, key, medians, rows) for concurrency in levels(rows)]
        panels.append(f'<section class="tab-panel" id="panel-{tab}"><div class="table-scroll"><table class="comparison-table"><thead><tr><th>Concurrency</th><th>Valid runs off / on</th><th class="av-off">AV off</th><th class="av-on">AV on</th><th class="delta">Δ on − off</th><th class="delta">Δ %</th></tr></thead><tbody>{"".join(body)}</tbody></table></div></section>')
    tables = "".join(choices) + f'<div class="metric-tabs">{"".join(labels)}</div><div class="panels">{"".join(panels)}</div>'
    modes = sorted({r["mode"] for r in rows})
    commands = {mode: public((source / f"{mode}{MODE_SUFFIX}" / "server-command.txt").read_text()) for mode in modes}
    shown = "".join(f"<h2>AV {mode}</h2><pre>{html.escape(cmd)}</pre>" for mode, cmd in commands.items())
    count = sum(r["cell"] == "main" for r in rows)
    valid = len(measured(rows))
    cells = (source / "cells.txt").read_text().strip()
    body = f'''<p><strong>{capture_note(commands[modes[0]])}</strong> {valid} of {count} measurement cells passed all validity gates (c:requests = <code>{cells}</code>, measured {run_date(source)}).</p>
<section class="card"><figure>{svg}<figcaption>Faint points show individual valid runs. Lines connect medians. Interactivity = 1000 / mean request TPOT in milliseconds.<br><strong>Download figure:</strong> <a href="{slug}/throughput-interactivity.png" download="qwen35-av-{slug}-throughput-interactivity.png">PNG</a> · <a href="{slug}/throughput-interactivity.svg" download="qwen35-av-{slug}-throughput-interactivity.svg">SVG</a></figcaption></figure></section>
<section class="card"><h2>Metrics</h2>{tables}<p class="metric-note">Values are medians across valid runs. Δ is AV on minus AV off, and Δ% is relative to AV off. <a href="{slug}/concurrency-summary.csv">Median CSV</a> · <a href="{slug}/summary.csv">Run-level CSV</a> · <a href="{slug}/summary.json">Median JSON</a> · <a href="{slug}/throughput-interactivity.svg">SVG</a> · <a href="{slug}/throughput-interactivity.png">PNG</a></p></section>
<details><summary>Server commands</summary>{shown}</details>'''
    return dict(slug=slug, label=label, source=source, html=body, rules=rules)


def run_date(source: Path) -> str:
    """Run dirs end in YYYYMMDD."""
    stamp = source.name.rsplit("-", 1)[-1]
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}"


def capture_note(command: str) -> str:
    """Describe the CUDA graph capture list from the server command."""
    if "cudagraph_capture_sizes" not in command:
        return "Default CUDA graph capture sizes (up to 1,024 tokens on B300): steps with a prefill exceed them and run eagerly."
    sizes = [int(s) for s in command.split("cudagraph_capture_sizes")[1].split("[")[1].split("]")[0].replace("\\", "").split(",")]
    extra = [s for s in sizes if s > 1024]
    return (f"Extended CUDA graph capture sizes: the default list plus {len(extra)} sizes from {extra[0]:,} to {extra[-1]:,} tokens, "
            "so steps with a prefill run with piecewise graphs.")


def tab_rules(tab: str, bar: str, panel: str) -> list[str]:
    return [f'#{tab}:checked~.{bar} label[for="{tab}"]{{background:#235cca;color:white}}',
            f'#{tab}:checked~{panel}{{display:block}}',
            f'#{tab}:focus-visible~.{bar} label[for="{tab}"]{{outline:3px solid #e4ad31;outline-offset:3px}}']


def comparison_row(concurrency: int, key: str, medians: list[dict], rows: list[dict]) -> str:
    points = {r["mode"]: r for r in medians if r["concurrency"] == concurrency}
    cells = {m: sum(r["cell"] == "main" and r["concurrency"] == concurrency and r["mode"] == m for r in rows) for m in ("off", "on")}
    runs = " / ".join(f'{points[m]["runs"] if m in points else 0}/{cells[m]}' for m in ("off", "on"))
    if len(points) < 2:
        return f'<tr><th scope="row">{concurrency}</th><td>{runs}</td><td colspan="4">no valid runs in at least one mode</td></tr>'
    off, on = points["off"][key], points["on"][key]
    delta = on - off
    percent = "N/A" if off == 0 else f"{100 * delta / off:+.2f}%"
    return f'<tr><th scope="row">{concurrency}</th><td>{runs}</td><td class="av-off">{off:,.2f}</td><td class="av-on">{on:,.2f}</td><td class="delta">{delta:+,.2f}</td><td class="delta">{percent}</td></tr>'


def records_commit(source: Path) -> str:
    return (source / "commit.txt").read_text().strip()


PUBLIC_PATHS = {
    "/home/tomasruizt/.venv/bin/python -m vllm.entrypoints.openai.api_server --model": "vllm serve",
    "/data/engine/hub_cache/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754": "Qwen/Qwen3.5-35B-A3B",
    "/data/tomasruizt/hub_cache/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754": "Qwen/Qwen3.5-35B-A3B",
    "/data/tomasruizt/benchmarks/deepseek-v4-dep4-av/dataset/throughput_1k.jsonl": "speed_bench_throughput_1k.jsonl",
}


def public(text: str) -> str:
    """Replace machine-local paths in reproduction commands with portable names."""
    for local, portable in PUBLIC_PATHS.items():
        text = text.replace(local, portable)
    return text


if __name__ == "__main__":
    main()
