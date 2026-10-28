"""Render the local validation summary with complete serving commands."""

import argparse
import html
import json
import shlex
import shutil
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit = json.loads((args.source / "audit.json").read_text())
    comparison = json.loads((args.source / "comparison.json").read_text())
    cells = "".join(
        f"<tr><td>{r['concurrency']}</td><td>{r['fixed_tok_s']:.0f}</td><td>{r['adaptive_tok_s']:.0f}</td><td>{r['gain_pct']:+.1f}%</td><td>{r['fixed_tpot_ms']:.2f}</td><td>{r['adaptive_tpot_ms']:.2f}</td></tr>"
        for r in comparison
    )
    correctness = "".join(
        f"<li>{mode}: {score['accuracy'] * 100:.0f}/100 GSM8K, invalid rate {score['invalid_rate']:.0%}.</li>"
        for mode, score in audit["correctness"]["scores"].items()
    )
    evidence = audit["execution"]
    software = (args.source / "software.json").read_text()
    aiperf = (args.source / "aiperf-version.txt").read_text().strip()
    answers = " ".join(
        f"{mode}: {len(row['answer_mismatches'])} numeric answer differences from the reference, {row['exact_text_matches']}/100 exact text matches, {row['length_capped']} length-capped outputs."
        for mode, row in audit["correctness"]["comparisons"].items()
    )
    answers += f" Fixed versus adaptive: {len(audit['correctness']['fixed_vs_adaptive']['answer_mismatches'])} numeric answer differences."
    commands = "".join(
        f"<details><summary>{mode} serving command</summary><pre>{html.escape(server_command(args.source / mode / 'command.json'))}</pre></details>"
        for mode in ("fixed", "adaptive")
    )
    for filename in ("audit.json", "metrics.csv", "comparison.json", "source.patch"):
        shutil.copyfile(args.source / filename, args.output / filename)
    figure = (
        args.source.parent.parent
        / "plots"
        / args.source.name
        / "throughput-interactivity.png"
    )
    shutil.copyfile(figure, args.output / figure.name)
    configs = "".join(
        benchmark_config(args.source, args.output, mode)
        for mode in ("fixed", "adaptive")
    )
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GDN adaptive verification, K=15</title><style>body{{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 24px;color:#17212b}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:right;padding:10px;border-bottom:1px solid #ddd}}pre{{white-space:pre-wrap;background:#f4f6f8;padding:16px;font-size:13px}}details{{margin:16px 0}}img{{max-width:100%}}a{{color:#1264a3}}</style>
<h1>GDN adaptive verification with K=15</h1><p>Qwen3.6-27B BF16 + satgeze/Qwen3.6-27B-DSpark (trained block size 15). Target query width 16. Two B300s, TP2, 2026-10-27.</p>
<h2>Correctness and execution</h2><ul>{correctness}<li>65 metadata tests passed.</li><li>Confidence head loaded and consumed.</li><li>{evidence["full_replays_with_gpu_layout"]} FULL varlen graph replays, {evidence["ragged_full_replay"]} genuinely ragged batches, {evidence["wide_full_replay"]} with query lengths above 8. Maximum observed query length: {evidence["max_replayed_query_len"]}.</li><li>Fused CUDA GDN disabled at width 16. Nsight recorded {evidence["triton_fallback_kernel_instances"]:,} Triton GDN kernel instances during real requests.</li></ul>
<p>{html.escape(answers)}</p><p>The staggered diagnostic workload also exercised {evidence["mixed_prefill_decode_steps"]} mixed prefill/decode steps. Example GPU query lengths during a FULL replay: <code>{evidence["ragged_example"]["query_lens"]}</code>.</p>
<h2>Performance</h2><p>Median of three AIPerf measurements per cell, using the steady window after initial completions and before drain. SPEED-Bench throughput_1k, natural EOS with output cap 1024, temperature 0, thinking off. Prefix caching disabled. Timed servers have no execution probes. Send durations at c1/16/64/128: fixed 30/30/45/60 seconds, adaptive 30/30/30/40 seconds, sized to retain usable steady windows.</p>
<table><tr><th>Concurrency</th><th>Fixed tok/s</th><th>Adaptive tok/s</th><th>Change</th><th>Fixed TPOT ms</th><th>Adaptive TPOT ms</th></tr>{cells}</table>
<p>One persistent server per arm, fixed followed by adaptive on the same two GPUs. These are within-session repeats, so startup variation and drift limit causal precision. The execution trace is from a separate instrumented TP2 server, with replay counts reported for rank 0 and kernel counts across ranks. A TP1 capacity check admitted only about 65 active requests at K=15, so TP2 is used consistently to support c128 without cache pressure.</p>
<img src="throughput-interactivity.png" alt="Throughput and interactivity comparison">
<h2>Reproduce</h2><p>Source: <a href="https://github.com/vllm-project/vllm/commit/b4a325b656b5ae293a0c733ac60a1e05901216dc">b4a325b656 (PR #60229 snapshot on October 26)</a>, plus <a href="https://github.com/tomasruizt/vllm/commit/1f61ebb0ae8760598c433af817ebedc668cc504f">1f61ebb0ae (remove graph capability cap)</a>, <a href="https://github.com/tomasruizt/vllm/commit/6cc628aed6">6cc628aed6 (gather simplification)</a>, and removal of the redundant capture conditional. Installed precompiled extensions were reused. <a href="source.patch">Exact patch relative to 1f61ebb0ae</a>.</p>
{commands}{configs}<p>Use the pinned target tokenizer and place the SPEED-Bench throughput_1k dataset at the configured path. Both arms use three repeats, grace period 120 seconds, and seed 42.</p>
<details><summary>Software versions</summary><pre>{html.escape(software)}\nAIPerf: {html.escape(aiperf)}</pre></details><h2>Artifacts</h2><p><a href="audit.json">Execution and correctness audit</a> · <a href="metrics.csv">All 24 measurements</a> · <a href="comparison.json">Performance comparison</a></p></html>"""
    (args.output / "index.html").write_text(body)


def server_command(
    path: Path,
    target_model: str = "Qwen/Qwen3.6-27B",
    target_revision: str = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    draft_model: str = "satgeze/Qwen3.6-27B-DSpark",
    draft_revision: str = "47d8de04dd814f4a727bf3fc6a85613d2a3dcaa1",
) -> str:
    command = json.loads(path.read_text())
    args = command[5:]
    if "--speculative-config" in args:
        spec_index = args.index("--speculative-config") + 1
        spec = json.loads(args[spec_index])
        spec["model"] = draft_model
        spec["revision"] = draft_revision
        args[spec_index] = json.dumps(spec)
    command = [
        "vllm",
        "serve",
        target_model,
        "--revision",
        target_revision,
        *args,
    ]
    return (
        "VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=1024 "
        + shlex.join(command)
    )


def benchmark_config(
    source: Path,
    output: Path,
    mode: str,
    tokenizer: str = "Qwen/Qwen3.6-27B",
) -> str:
    config = yaml.safe_load((source / mode / "sweep.yaml").read_text())
    config["benchmark"]["tokenizer"]["name"] = tokenizer
    config["benchmark"]["datasets"][0]["path"] = "throughput_1k.jsonl"
    config["benchmark"]["artifacts"]["dir"] = f"artifacts/{mode}"
    text = yaml.safe_dump(config, sort_keys=False)
    (output / f"{mode}-sweep.yaml").write_text(text)
    return f'<pre>TMPDIR=/tmp aiperf profile --config {mode}-sweep.yaml</pre><details><summary>{mode} benchmark configuration (<a href="{mode}-sweep.yaml">download</a>)</summary><pre>{html.escape(text)}</pre></details>'


if __name__ == "__main__":
    main()
