"""Regenerate the comparison tables from the preserved AIPerf exports."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    lines = [
        "# DFlash comparison",
        "",
        "One H100, TP=1, FP8 weights, GSM8K, up to 256 output tokens, EOS enabled, 15 proposed tokens (SGLang block size 16).",
        "",
        "Main/PR/earlier PR2: 92% memory, BF16 GDN states, prefix caching enabled. Current SGLang: 95%, BF16 states, prefix caching disabled, prefill chunk 2,048; server cap 32 for 4B and 16 for 27B. Configuration differences remain in engine comparisons.",
        "",
        "PR = #57962; PR2 = #52297 merged onto the same main base. New PR2 c=1,7,16,32 sweeps with prefix caching disabled were queued at migration time; no completed results were available. Existing PR2 values below are the earlier runs.",
        "",
    ]
    for model in ("4B", "27B"):
        for metric, title, precision in (
            ("output_token_throughput", "output tok/s", 1),
            ("inter_chunk_latency", "ITL p99 (ms)", 2),
        ):
            lines += [
                f"## {model} {title}",
                "",
                "| Concurrency | Main | PR | PR vs main | PR2 | PR2 vs main | SGLang | active reqs |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
            ]
            for concurrency in (1, 7, 8, 16, 32):
                runs = {
                    engine: read_run(model, engine, concurrency)
                    for engine in ("main", "pr", "pr2", "sglang")
                }
                stat = "p99" if metric == "inter_chunk_latency" else "avg"
                values = {e: r[0][metric][stat] if r else None for e, r in runs.items()}
                fmt = lambda v, precision=precision: (
                    "—" if v is None else f"{v:,.{precision}f}"
                )
                delta = lambda e, values=values: (
                    "—"
                    if values[e] is None or values["main"] is None
                    else f"{100 * (values[e] / values['main'] - 1):+.1f}%"
                )
                discrepancies = [
                    f"{label}: {r[1]:g}"
                    for e, label in (
                        ("main", "Main"),
                        ("pr", "PR"),
                        ("pr2", "PR2"),
                        ("sglang", "SGLang"),
                    )
                    if (r := runs[e]) and r[1] != concurrency
                ]
                active = "; ".join(discrepancies) or (
                    "Same as concurrency" if any(runs.values()) else "Not measured"
                )
                lines.append(
                    f"| {concurrency} | {fmt(values['main'])} | {fmt(values['pr'])} | {delta('pr')} | {fmt(values['pr2'])} | {delta('pr2')} | {fmt(values['sglang'])} | {active} |"
                )
            lines += [""]
    lines += [
        "Active reqs show observed maxima only where they differ from client concurrency. SGLang's gauge occasionally exceeds client concurrency by one; values are preserved as reported.",
        "",
        "Measured/warmup requests: c=1 100/10, c=7 140/14 (pending), c=8 160/16, c=16 320/32, c=32 640/64. Single runs; no uncertainty estimates.",
        "",
        "Sources: [main/PR/PR2 artifacts](results/pr57962-sweep/), [current SGLang artifacts](results/sglang-no-prefix-95-bf16-prefill2048/), [revisions](revisions.json). Regenerate with `python scripts/render_results.py`.",
        "",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))


def read_run(model, engine, concurrency):
    if engine == "sglang":
        path = (
            ROOT
            / "results/sglang-no-prefix-95-bf16-prefill2048"
            / model
            / "sglang_dflash"
            / f"c{concurrency}"
        )
        key = "sglang:num_running_reqs"
    else:
        path = (
            ROOT
            / "results/pr57962-sweep"
            / model
            / engine
            / f"c{concurrency}"
            / "vllm_dflash"
        )
        key = "vllm:num_requests_running"
    if not (path / "summary.json").exists():
        return None
    report = json.loads((path / "aiperf/profile_export_aiperf.json").read_text())
    metrics = json.loads((path / "aiperf/server_metrics_export.json").read_text())
    return report, metrics["metrics"][key]["series"][0]["stats"]["max"]


if __name__ == "__main__":
    main()
