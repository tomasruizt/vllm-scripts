"""Render the B200 release comparison from completed AIPerf exports."""

import argparse
import html
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

VARIANTS = {
    "vllm_baseline": "vLLM",
    "vllm_dflash": "vLLM DFlash",
    "sglang_baseline": "SGLang",
    "sglang_dflash": "SGLang DFlash",
    "pr2_dflash": "vLLM + DFlash PR 52297",
}
CONCURRENCIES = (1, 2, 4, 8, 16, 32)
MODELS = {"27B": "Qwen3.5-27B", "4B": "Qwen3.5-4B", "35B-A3B": "Qwen3.6-35B-A3B"}
METRICS = (
    ("output_token_throughput", "avg", "output tok/s", "tok/s"),
    ("inter_chunk_latency", "p99", "ITL p99 (ms)", "ITL p99"),
    ("time_to_first_token", "p99", "TTFT p99 (ms)", "TTFT p99"),
    ("inter_token_latency", "p90", "TPOT p90 (ms)", "TPOT p90"),
    ("acceptance_length", "avg", "Acceptance length (including bonus)", "Acceptance length"),
)
CAPACITY_METRICS = ("Reported cache tokens", "128K seq equivalents (est.)", "Configured request limit")
CAPACITY_NOTE = "128K = 131,072 tokens. Sequence equivalents = reported cache tokens / 131,072, not measured concurrency. These allocations come from servers configured for 32,768-token contexts and 32 request slots; reconfiguring for 128K may change capacity. No long-context measurements were run."
LOG_NOTE = "Each table's Logs link opens server and benchmark logs, metric exports, and configurations for every repetition."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--num-speculative-tokens", type=int)
    parser.add_argument("--repeats", type=Path, help="Completed campaign containing report_metrics.csv and capacity_metrics.csv")
    args = parser.parse_args()
    if args.num_speculative_tokens is None:
        manifest = args.root / "experiment.json"
        args.num_speculative_tokens = json.loads(manifest.read_text())["num_speculative_tokens"] if manifest.exists() else 15
    models = [model for model in MODELS if (args.root / model).exists()]
    render_benchmark_logs(args.root, models, args.repeats)
    repetitions = 3 if args.repeats else 1
    expected = len(models) * len(VARIANTS) * len(CONCURRENCIES) * repetitions
    lines = [
        f"# B200 release comparison — DFlash block {args.num_speculative_tokens + 1}",
        "",
        "[Open the tabbed HTML report](RESULTS.html).",
        "",
        LOG_NOTE,
        "",
        "- **Releases:** vLLM 0.30.0 (`ced6857afa0ea7b2e3f0846a62e1394e90f15607`); SGLang 0.5.20. Latest published releases checked on 2026-09-22.",
        "- **PR2:** head `a8db1fe32ac19c2296bc0e9faedf9550ee56dd2d` merged with vLLM 0.30.0 as `63dc18cf5a93f69be959b2d2f3c26109ac693766`. Only PR2's four files differ; compiled kernels come from the release wheel.",
        "",
        "- **Hardware and precision:** one NVIDIA B200 per run, TP=1, FP8 weights, BF16 compute and BF16 Mamba convolution/SSM states.",
        "- **Server limits:** prefix caching disabled; memory fraction 0.92; 32 request slots; prefill chunk 2,048; maximum context 32,768.",
        "- **CUDA graphs:** vLLM uses MRV2 and capture sizes through 512 tokens; SGLang uses its release-default graph policy.",
        f"- **DFlash:** {args.num_speculative_tokens} proposed tokens (SGLang block size {args.num_speculative_tokens + 1}); ReplaySSM is not enabled explicitly.",
        "- **Acceptance length:** includes the bonus token. vLLM/PR use 1 + accepted draft tokens / draft iterations; SGLang uses the mean sampled acceptance-length gauge, with different weighting. Baselines are not applicable (—).",
        f"- **MoE comparison:** Qwen3.6-35B-A3B uses the same workload and {args.num_speculative_tokens} proposals as this matrix. The PR's original H200 experiment used ShareGPT and 8 proposals; this is not an exact reproduction.",
        "- **MoE backend:** both SGLang MoE variants use Triton because release-default TRTLLM failed with missing FP8 input scales in the original block-16 sweep.",
        "- **Draft backend:** vLLM and PR2 explicitly use `TRITON_ATTN`; SGLang also selects Triton. The original block-16 sweep's vLLM release-default FlashAttention draft backend failed on B200 with `No common block size for 336/480`.",
        "",
        "- **Workload:** GSM8K, up to 256 output tokens, EOS and model sampling defaults enabled. Dense-model checkpoints match the pinned H100 experiments; all target/draft revisions are recorded per run.",
        "- **Requests/warmups:** 100/10 at c=1, otherwise 20c/2c. Each variant reuses one server across ascending concurrency levels.",
        "- **Execution:** variants run concurrently on separate reserved GPUs on a shared host, using up to seven benchmark GPUs.",
        ("- **Scope:** n=3 independently restarted runs per configuration; tables show means. Latencies average per-run percentiles: TPOT p90, ITL and TTFT p99, not pooled percentiles. Plot coordinates average each run's throughput and 1,000 / TPOT p90 separately; error bars show ±1 sample SD."
         if args.repeats else "- **Scope:** single runs without uncertainty estimates; reported separately from the historical H100 tables."),
        *(["- **Concurrency caveat:** C=1 measures GSM8K prompt indices 10–109, whereas C=2 measures 4–43 (zero-based), because request and warmup counts differ. These curves retain the measured workloads; n=3 does not correct the prompt-mix confound. Small request counts limit p99 reliability."] if args.repeats else []),
        "",
    ]
    overview = list(lines)
    if (args.root / "SGLANG_VERIFICATION.md").exists():
        lines += ["[SGLang verification findings](SGLANG_VERIFICATION.md).", ""]
    all_runs = {}
    completed = 0
    for model in models:
        runs = {
            (variant, c): read_run(args.root, model, variant, c, args.repeats)
            for variant in VARIANTS
            for c in CONCURRENCIES
        }
        all_runs[model] = runs
        headings = list(VARIANTS.values())
        logs_row = "| Logs | " + " | ".join(log_links(args.root, model, variant, markdown=True)
                                             for variant in VARIANTS) + " |"
        completed += sum(run is not None for run in runs.values()) * repetitions
        for metric, stat, title, _ in METRICS:
            lines += [
                f"## {model} {title}",
                "",
                "| c | " + " | ".join(headings) + " |",
                "| ---: | " + " | ".join("---:" for _ in VARIANTS) + " |",
            ]
            for c in CONCURRENCIES:
                values = []
                for variant in VARIANTS:
                    run = runs[variant, c]
                    values.append(format_metric(run, metric, stat))
                lines.append(f"| {c} | " + " | ".join(values) + " |")
            lines += [logs_row, ""]
        lines += [
            f"## {model} Memory & capacity", "",
            "| Metric | " + " | ".join(headings) + " |",
            "| --- | " + " | ".join("---:" for _ in VARIANTS) + " |",
        ]
        for row in read_capacity(args.root, model, args.repeats):
            lines.append("| " + " | ".join(row) + " |")
        lines += [logs_row, "", CAPACITY_NOTE, ""]
    lines += [
        f"- **Completed:** {completed}/{expected} benchmark points. A dash indicates an unavailable or inapplicable metric.",
        "- **Artifacts:** per-run commands, versions, GPU identifiers, metrics and logs are in the model/variant directories.",
        "- **Environment:** see `environment.json` for dependency versions and source revisions.",
        "",
    ]
    for model in all_runs:
        plot_path = f"plots/{model.lower()}-throughput-interactivity.svg"
        if not (args.root / plot_path).exists():
            continue
        lines += [
            f"## {model} throughput vs interactivity",
            "",
            f"![{model} throughput versus interactivity]({plot_path})",
            "",
            ("Coordinates: mean throughput and mean per-run 1,000 / TPOT p90 (ms); error bars: ±1 sample SD, n=3. Dashed curves show the baselines." if args.repeats else "Interactivity = 1,000 / TPOT p90 (ms). Dashed curves show the baselines."),
            "",
        ]
    (args.root / "RESULTS.md").write_text("\n".join(lines))
    render_html(args.root, all_runs, overview, completed, expected, args.num_speculative_tokens + 1, args.repeats)
    print(f"Completed {completed}/{expected} points; wrote {args.root / 'RESULTS.md'}")


def render_html(root, all_runs, overview, completed, expected, block_size, repeats=None):
    controls, tabs, panels, tab_styles = [], [], [], []
    for index, model in enumerate(all_runs):
        runs = all_runs[model]
        key = f"model-{model.lower()}"
        control, tab, styles = tab_control(key, model, "model", index == 0)
        controls.append(control)
        tabs.append(tab)
        tab_styles.extend(styles)
        panel, metric_styles = render_model_panel(root, model, runs, repeats)
        panels.append(panel)
        tab_styles.extend(metric_styles)
    notes = "\n".join(
        "<li>" + html.escape(line[2:].replace("**", "").replace("`", "")) + "</li>"
        for line in overview if line.startswith("- ")
    )
    template = Path(__file__).with_name("results_template.html").read_text()
    for key, value in {
        "__TABS__": "\n".join(tabs), "__PANELS__": "\n".join(panels),
        "__CONTROLS__": "\n".join(controls), "__TAB_STYLES__": "\n".join(tab_styles),
        "__NOTES__": notes, "__COMPLETED__": str(completed),
        "__EXPECTED__": str(expected),
        "__BLOCK_SIZE__": str(block_size),
        "__STATISTICS_NOTE__": ("n=3 · Tables show means; latency tables average per-run percentiles (TPOT p90; ITL and TTFT p99)." if repeats else "Single runs; no uncertainty estimates."),
    }.items():
        template = template.replace(key, value)
    (root / "RESULTS.html").write_text(template)


def render_model_panel(root, model, runs, repeats=None):
    model_key = f"model-{model.lower()}"
    headings = [html.escape(label) for label in VARIANTS.values()]
    logs_row = ('<tfoot><tr><th scope="row">Logs</th>'
                + ''.join(f'<td>{log_links(root, model, variant)}</td>' for variant in VARIANTS)
                + '</tr></tfoot>')
    controls, tabs, panels, styles = [], [], [], []
    for index, (metric, stat, title, label) in enumerate(METRICS):
        key = f"metric-{model.lower()}-{index}"
        control, tab, tab_styles = tab_control(key, label, f"metric-{model.lower()}", index == 0)
        controls.append(control)
        tabs.append(tab)
        styles.extend(tab_styles)
        source_note = {
            "inter_chunk_latency": "Source: Metrics JSON → inter_chunk_latency.p99 (not printed in the console). Measured between streamed chunks, which may contain multiple tokens. Follow Logs for the export.",
            "inter_token_latency": "Source: Metrics JSON → inter_token_latency.p90, called Inter Token Latency in the AIPerf console. Follow Logs for the export.",
            "acceptance_length": "Source: Server metrics JSON. vLLM/PR: 1 + accepted draft tokens / draft iterations; SGLang: sampled spec_accept_length gauge average. Includes the bonus token; aggregation differs between engines. Follow Logs for exports and formulas.",
        }.get(metric, "")
        parts = [
            f'<section class="tab-panel" id="panel-{key}" aria-labelledby="label-{key}">',
            f'<p class="metric-note">{html.escape(title)} · '
            f'{"Higher" if stat == "avg" else "Lower"} is better</p>',
            f'<p class="metric-note">{html.escape(source_note)}</p>' if source_note else '',
            '<div class="table-scroll"><table>',
            '<thead><tr><th scope="col">Concurrency</th>',
        ]
        parts += [f'<th scope="col">{heading}</th>' for heading in headings]
        parts.append("</tr></thead><tbody>")
        for c in CONCURRENCIES:
            parts.append(f'<tr><th scope="row">{c}</th>')
            for variant in VARIANTS:
                run = runs[variant, c]
                cell = format_metric(run, metric, stat)
                parts.append(f'<td>{cell}</td>')
            parts.append("</tr>")
        parts.append(f"</tbody>{logs_row}</table></div></section>")
        panels.append("\n".join(parts))
    key = f"metric-{model.lower()}-capacity"
    control, tab, tab_styles = tab_control(key, "Memory & capacity", f"metric-{model.lower()}", False)
    controls.append(control)
    tabs.append(tab)
    styles.extend(tab_styles)
    parts = [
        f'<section class="tab-panel" id="panel-{key}" aria-labelledby="label-{key}">',
        f'<p class="metric-note">{html.escape(CAPACITY_NOTE)}</p>',
        '<div class="table-scroll"><table><thead><tr>',
        '<th scope="col">Metric</th>',
        *[f'<th scope="col">{heading}</th>' for heading in headings],
        '</tr></thead><tbody>',
    ]
    for row in read_capacity(root, model, repeats):
        parts.append('<tr><th scope="row">' + html.escape(row[0]) + '</th>'
                     + ''.join(f'<td>{html.escape(value)}</td>' for value in row[1:]) + '</tr>')
    parts.append(f'</tbody>{logs_row}</table></div></section>')
    panels.append('\n'.join(parts))
    plot_path = root / f"plots/{model.lower()}-throughput-interactivity.svg"
    plot = ""
    if plot_path.exists():
        svg = plot_path.read_text()
        # Each inline Matplotlib SVG otherwise reuses IDs such as figure_1.
        for svg_id in set(re.findall(r'\bid="([^"]+)"', svg)):
            unique_id = f"{model_key}-{svg_id}"
            svg = svg.replace(f'id="{svg_id}"', f'id="{unique_id}"')
            svg = svg.replace(f'#{svg_id}"', f'#{unique_id}"')
            svg = svg.replace(f'#{svg_id})', f'#{unique_id})')
        downloads = []
        for extension in ("png",):
            asset = plot_path.with_suffix(f".{extension}")
            if asset.exists():
                href = html.escape(asset.relative_to(root).as_posix(), quote=True)
                filename = html.escape(f"{root.name}-{asset.name}", quote=True)
                downloads.append(f'<a href="{href}" download="{filename}">{extension.upper()}</a>')
        caption = '<figcaption>Download plot: ' + ' · '.join(downloads) + '</figcaption>'
        plot = f'<figure aria-label="{model} throughput versus interactivity">' + svg[svg.index("<svg"):] + caption + "</figure>"
    panel = [
        f'<section class="tab-panel" id="panel-{model_key}" aria-labelledby="label-{model_key}">',
        f'<h2>{MODELS[model]}</h2>',
        plot,
        f'<div class="metric-group" role="group" aria-label="{model} benchmark metric">',
        *controls, '<div class="metric-tabs">', *tabs, '</div>',
        '<div class="panels">', *panels, '</div></div>', '</section>',
    ]
    return "\n".join(panel), styles


def tab_control(key, label, group, selected):
    checked = " checked" if selected else ""
    control = (
        f'<input class="metric-choice" type="radio" name="{group}" '
        f'id="{key}" aria-controls="panel-{key}"{checked}>'
    )
    tab = f'<label id="label-{key}" for="{key}">{html.escape(label)}</label>'
    styles = [
        f'#{key}:checked ~ .panels > #panel-{key} {{ display: block; }}',
        f'#{key}:checked ~ .metric-tabs > label[for="{key}"] '
        '{ background: #235cca; color: white; }',
        f'#{key}:focus-visible ~ .metric-tabs > label[for="{key}"] '
        '{ outline: 3px solid #e4ad31; outline-offset: 3px; }',
    ]
    return control, tab, styles


def render_benchmark_logs(root, models, repeats=None):
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>Benchmark logs</title><style>body{font-family:system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 24px;color:#111827}a{color:#225ac6}li{margin:12px 0}section{margin:36px 0}</style></head><body>',
        '<a href="RESULTS.html">Back to report</a><h1>Benchmark logs and metric sources</h1>',
        '<p>Each concurrency links to the original AIPerf exports used by the report. Values in console logs are rounded; some metrics are not printed.</p>',
        '<ul>',
        '<li><b>Output tok/s:</b> Metrics JSON → <code>output_token_throughput.avg</code>.</li>',
        '<li><b>ITL p99:</b> Metrics JSON → <code>inter_chunk_latency.p99</code> (ms). This is the gap between streamed content chunks, which can contain multiple tokens. It is not printed in the benchmark console table.</li>',
        '<li><b>TTFT p99:</b> Metrics JSON → <code>time_to_first_token.p99</code> (ms).</li>',
        '<li><b>TPOT p90:</b> Metrics JSON → <code>inter_token_latency.p90</code> (ms). The console calls this “Inter Token Latency”; the report labels it TPOT.</li>',
        '<li><b>Pareto plot:</b> x = <code>1000 / inter_token_latency.p90</code>; y = <code>output_token_throughput.avg</code>.</li>',
        '<li><b>vLLM / PR 52297 AL:</b> Server metrics JSON → <code>1 + sum(metrics["vllm:spec_decode_num_accepted_tokens"].series[*].stats.total) / sum(metrics["vllm:spec_decode_num_drafts"].series[*].stats.total)</code>.</li>',
        '<li><b>SGLang AL:</b> Server metrics JSON → <code>sum(metrics["sglang:spec_accept_length"].series[*].stats.avg)</code> (one series for these TP1 runs). Both AL values include the bonus token; the SGLang sampled gauge average has different weighting from the vLLM counter ratio. Summary JSON records AL and its method; baselines have no AL.</li>',
        '<li><b>Cache capacity:</b> Server log → vLLM <code>GPU KV cache size</code> or SGLang <code>max_total_num_tokens</code>. Divide by <code>131072</code> for 128K sequence equivalents. Config JSON → <code>server_command</code> records the configured request limit.</li>',
        '</ul><p>These exports allow checking the reported aggregates and AL arithmetic. Independently recomputing percentiles from individual requests would additionally require the per-request traces, which are not included in this report bundle.</p>',
    ]
    for model in models:
        for variant, label in VARIANTS.items():
            parts += [f'<section id="{model}-{variant}"><h2>{html.escape(MODELS[model])} · {html.escape(label)}</h2><ul>']
            for source_index, source_root in enumerate(repetition_roots(root, repeats), 1):
                if repeats:
                    parts.append(f'<li><b>Repetition {source_index}</b></li>')
                    mode = "vllm_dflash" if variant == "pr2_dflash" else variant
                    server = Path(os.path.relpath(source_root / model / variant / mode / "server.log", root))
                    parts.append(f'<li><a href="{html.escape(server.as_posix(), quote=True)}">Server log</a></li>')
                for c in CONCURRENCIES:
                    engine_mode = "vllm_dflash" if variant == "pr2_dflash" else variant
                    run = Path(os.path.relpath(source_root / model / variant / engine_mode / f"c{c}", root))
                    links = [f'<a href="{html.escape((run / "benchmark.log").as_posix(), quote=True)}">Log</a>']
                    for filename, title in (
                        ("aiperf/profile_export_aiperf.json", "Metrics JSON"),
                        ("aiperf/server_metrics_export.json", "Server metrics JSON"),
                        ("summary.json", "Summary JSON"),
                        ("run_config.json", "Config JSON"),
                    ):
                        path = run / filename
                        if (root / path).is_file():
                            links.append(f'<a href="{html.escape(path.as_posix(), quote=True)}">{title}</a>')
                    parts.append(f'<li>Concurrency {c}: ' + ' · '.join(links) + '</li>')
            parts.append('</ul></section>')
    parts.append('</body></html>')
    if repeats:
        prefix = Path(os.path.relpath(repeats, root)).as_posix()
        parts.insert(-1, f'<p>Aggregates use all three repetitions. Tables: means of per-run statistics. Plots: mean(1000 / TPOT p90) and mean(throughput), with sample SD on both axes. <a href="{prefix}/report_metrics.csv">Long-format source measurements</a> · <a href="{prefix}/report_comparison.csv">Aggregate statistics</a> · <a href="{prefix}/capacity_metrics.csv">Capacity measurements</a></p>')
    (root / "benchmark-logs.html").write_text('\n'.join(parts))


def log_links(root, model, variant, *, markdown=False):
    path = f"benchmark-logs.html#{model}-{variant}"
    return f"[Logs]({path})" if markdown else f'<a href="{path}">Logs</a>'


def log_link(root, model, variant, label, concurrency=None, *, markdown=False):
    engine_mode = "vllm_dflash" if variant == "pr2_dflash" else variant
    path = Path(model) / variant / engine_mode
    path = path / "server.log" if concurrency is None else path / f"c{concurrency}" / "benchmark.log"
    if not (root / path).is_file():
        return label if markdown else html.escape(label)
    if markdown:
        return f"[{label}]({path.as_posix()})"
    kind = "Server" if concurrency is None else f"c={concurrency} benchmark"
    title = f"{model} {VARIANTS[variant]} — {kind} log"
    return f'<a href="{html.escape(path.as_posix(), quote=True)}" title="{html.escape(title, quote=True)}">{html.escape(label)}</a>'


def read_run(root, model, variant, concurrency, repeats=None):
    if repeats:
        frame = aggregate_metrics(repeats).loc[(block_size(root), model, variant, concurrency)]
        return {metric: {stat: frame.loc[metric, "mean"], "std": frame.loc[metric, "std"]}
                for metric, stat, *_ in METRICS if metric in frame.index}
    engine_mode = "vllm_dflash" if variant == "pr2_dflash" else variant
    path = root / model / variant / engine_mode / f"c{concurrency}"
    if not (path / "summary.json").exists():
        return None
    run = json.loads((path / "aiperf/profile_export_aiperf.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    run["acceptance_length"] = {"avg": summary.get("acceptance_length")}
    return run


def read_capacity(root, model, repeats=None):
    if repeats:
        frame = aggregate_capacity(repeats).loc[(block_size(root), model)]
        return [(metric, *(f'{frame.loc[(variant, metric), "mean"]:,.2f}'
                           for variant in VARIANTS)) for metric in CAPACITY_METRICS]
    rows = []
    for variant, label in VARIANTS.items():
        engine_mode = "vllm_dflash" if variant == "pr2_dflash" else variant
        path = root / model / variant / engine_mode
        log = (path / "server.log").read_text() if (path / "server.log").exists() else ""
        pattern = r"max_total_num_tokens=(\d+)" if variant.startswith("sglang") else r"GPU KV cache size: ([\d,]+) tokens"
        match = re.search(pattern, log)
        tokens = int(match[1].replace(",", "")) if match else None
        config_path = path / "c1/run_config.json"
        limit = "—"
        if config_path.exists():
            command = json.loads(config_path.read_text())["server_command"]
            flag = "--max-running-requests" if variant.startswith("sglang") else "--max-num-seqs"
            limit = command[command.index(flag) + 1]
        rows.append((label, f"{tokens:,}" if tokens is not None else "—",
                     f"{tokens / 131072:.2f}" if tokens is not None else "—", limit))
    return [(metric, *(row[index + 1] for row in rows))
            for index, metric in enumerate(CAPACITY_METRICS)]


def format_metric(run, metric, stat):
    data = run.get(metric, {}) if run else {}
    value = data.get(stat)
    if value is None:
        return "—"
    return f"{value:,.2f}"


def repetition_roots(root, repeats):
    return [root, *(repeats / f"rep{i}-block{block_size(root)}" for i in (2, 3))] if repeats else [root]


def block_size(root):
    manifest = root / "experiment.json"
    return json.loads(manifest.read_text())["num_speculative_tokens"] + 1 if manifest.exists() else 16


@lru_cache
def aggregate_metrics(repeats):
    frame = pd.read_csv(repeats / "report_metrics.csv")
    frame = frame.loc[
        (~frame.metric.isin(["inter_token_latency", "interactivity"]))
        | (frame.metric.eq("inter_token_latency") & frame.statistic.eq("p90"))
        | (frame.metric.eq("interactivity") & frame.statistic.eq("derived_p90"))
    ]
    keys = ["block", "model", "variant", "concurrency", "metric"]
    if frame.duplicated(keys + ["repetition"]).any():
        raise ValueError("Duplicate benchmark measurements")
    result = frame.groupby(keys).value.agg(["mean", "std", "count"])
    if not result["count"].eq(3).all() or not set(frame.repetition) == {1, 2, 3}:
        raise ValueError("Expected three complete repetitions for every metric")
    return result


@lru_cache
def aggregate_capacity(repeats):
    frame = pd.read_csv(repeats / "capacity_metrics.csv")
    result = frame.groupby(["block", "model", "variant", "metric"]).value.agg(["mean", "std", "count"])
    if not result["count"].eq(3).all():
        raise ValueError("Expected three capacity measurements per configuration")
    return result


if __name__ == "__main__":
    main()
