"""Collect benchmark JSON and Prometheus samples into one per-run Parquet."""

import argparse
from functools import cache
import json
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq
from prometheus_client.parser import text_string_to_metric_families

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE)
    parser.add_argument("--output", type=Path, default=HERE / "metrics.parquet")
    args = parser.parse_args()
    root = args.source.resolve()
    metadata = json.loads((root / "result.json").read_text())
    rows = smoke_rows(root, metadata)
    for folder in sorted((root / "results").iterdir()):
        if (folder / "sweep.json").is_file():
            rows.extend(sweep_rows(root, folder, metadata))
        if (folder / "aiperf-sweep.json").is_file():
            rows.extend(aiperf_rows(root, folder, metadata))
        if (folder / "quick-result.json").is_file():
            rows.extend(quick_rows(folder))
    assert len({r["run_id"] for r in rows}) == len(rows), "Duplicate run IDs"
    columns = sorted(set().union(*(r.keys() for r in rows)))
    records = [{key: row.get(key) for key in columns} for row in rows]
    table = pa.Table.from_pylist(records)
    details = dict(table.schema.metadata or {})
    details[b"experiment_metadata"] = json.dumps(metadata, sort_keys=True).encode()
    details[b"description"] = b"One row per benchmark JSON. phase separates measurements, pilots and initial smoke. Prometheus samples preserve before/after values and raw differences, including gauges. Measurement snapshots also include the CLI initial request check. Smoke snapshots are cumulative after-values only. Missing fields remain null."
    table = table.replace_schema_metadata(details)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output, compression="zstd")
    assert pq.read_table(args.output).num_rows == len(rows)
    print(f"{args.output}: {len(rows)} runs, {table.num_columns} columns")


def sweep_rows(root: Path, folder: Path, metadata: dict) -> list[dict]:
    manifest = json.loads((folder / "sweep.json").read_text())
    assert manifest["status"] == "completed", folder
    measured = {r["result"]: r for r in manifest["runs"]}
    rows = []
    for mode in manifest["mode_order"]:
        files = sorted((folder / mode).glob("*.json"))
        files = [p for p in files if re.fullmatch(r"(?:pilot-)?c\d+(?:-r\d+)?", p.stem)]
        for path in files:
            rows.append(sweep_row(root, folder, path, measured, metadata, manifest))
    assert sum(r["phase"] == "measurement" for r in rows) == len(measured)
    return rows


def quick_rows(folder: Path) -> list[dict]:
    rows = json.loads((folder / "quick-result.json").read_text())
    for row in rows:
        row.update(spec_decode_acceptance_length=row["acceptance_length"],
                   output_throughput=row["throughput"], duration=row["duration_s"],
                   interactivity_tok_s_user=1000 / row["mean_tpot_ms"],
                   completed=row["requests"] if row["error_rate"] == 0 else None)
    return rows


def aiperf_rows(root: Path, folder: Path, metadata: dict) -> list[dict]:
    manifest = json.loads((folder / "aiperf-sweep.json").read_text())
    return [aiperf_row(root, folder, metadata, manifest, run) for run in manifest["runs"]]


def aiperf_row(root: Path, folder: Path, metadata: dict, manifest: dict, run: dict) -> dict:
    path = folder / run["source"]
    summary = json.loads(path.read_text())
    label = path.parent.name
    command = json.loads((path.parent.parent / f"{label}-command.json").read_text())
    server = json.loads((path.parent.parent / "server-command.json").read_text())
    requests = [json.loads(line) for line in path.with_name("profile_export.jsonl").read_text().splitlines()]
    data = dict(max_concurrency=run["concurrency"], mean_tpot_ms=run["mean_tpot_ms"],
                spec_decode_acceptance_length=run["acceptance_length"], duration=run["duration_s"],
                completed=run["requests"], failed=0, output_throughput=run["throughput"],
                total_input_tokens=int(summary["input_sequence_length"]["sum"]),
                total_output_tokens=int(summary["output_sequence_length"]["sum"]))
    row = common_row(root, path, data, metadata)
    row.update(suite=folder.name, phase=run["phase"], av_enabled=run["mode"] == "on",
               repeat=run["repeat"], execution_order=manifest["runs"].index(run) + 1,
               server_session=f"{folder.name}-{run['mode']}", commit=manifest["commit"],
               gpu_index=manifest["gpu"], seed=42, command=command, server_command=server,
               dataset="nvidia/SPEED-Bench/throughput_1k", dataset_sha256=manifest["dataset_sha256"],
               workload_sha256=run["workload_sha256"], temperature=0.0, output_len=1024,
               natural_eos=True, thinking=False, prefix_caching=False,
               mean_verified_width=run["mean_verified_width"], benchmark_client="AIPerf 0.13.0",
               max_num_seqs=int(option(server, "--max-num-seqs")),
               max_num_batched_tokens=int(option(server, "--max-num-batched-tokens")),
               max_model_len=int(option(server, "--max-model-len")),
               instrumentation="PR58340 runtime metrics only, no per-step logging",
               prometheus_scope="AIPerf invocation including warmup in pilots",
               aiperf_summary_json=json.dumps(summary, separators=(",", ":")),
               aiperf_requests=[aiperf_request(record) for record in requests])
    for prefix, metric in (("ttft", "time_to_first_token"), ("tpot", "inter_token_latency"),
                           ("itl", "inter_chunk_latency"), ("e2el", "request_latency")):
        for name, statistic in (("mean", "avg"), ("median", "p50"), ("p90", "p90"), ("p99", "p99")):
            row[f"{name}_{prefix}_ms"] = summary[metric].get(statistic)
    profiled = [record for record in row["aiperf_requests"] if record["phase"] == "profiling"]
    row["output_cap_hit_fraction"] = sum(r["output_tokens"] >= 1024 for r in profiled) / len(profiled)
    outputs = path.with_name("outputs.json")
    row["aiperf_outputs_json"] = outputs.read_text() if outputs.exists() else None
    row["prometheus"] = samples(path.parent.parent / f"{label}-before.txt", path.parent.parent / f"{label}-after.txt")
    return row


def aiperf_request(record: dict) -> dict:
    metrics = record["metrics"]
    value = lambda key: metrics.get(key, {}).get("value")
    return dict(request_id=record["metadata"]["x_request_id"],
                conversation_id=record["metadata"]["conversation_id"],
                phase=record["metadata"]["benchmark_phase"],
                input_tokens=value("input_sequence_length"), output_tokens=value("output_sequence_length"),
                ttft_ms=value("time_to_first_token"), tpot_ms=value("inter_token_latency"),
                itl_ms=value("inter_chunk_latency"), metadata_json=json.dumps(record["metadata"]),
                metrics_json=json.dumps(metrics))


def sweep_row(root: Path, folder: Path, path: Path, measured: dict, metadata: dict, manifest: dict) -> dict:
    data = json.loads(path.read_text())
    command = json.loads(path.with_name(path.stem + "-command.json").read_text())
    server = json.loads((path.parent / "server-command.json").read_text())
    relative = str(path.relative_to(folder))
    run = measured.get(relative)
    row = common_row(root, path, data, metadata)
    row.update(suite=folder.name, phase="measurement" if run else "pilot",
               av_enabled=path.parent.name == "on", repeat=run["repeat"] if run else None,
               execution_order=manifest["runs"].index(run) + 1 if run else None,
               seed=int(option(command, "--seed")), command=command, server_command=server,
               gpu_index=manifest["gpu"], commit=manifest["commit"],
               server_session=manifest.get("server_session"),
               temperature=float(option(command, "--temperature")),
               dataset=option(command, "--hf-name") if "--hf-name" in command else option(command, "--dataset-name"),
               dataset_path=option(command, "--dataset-path") if "--dataset-path" in command else None,
               input_len=int(option(command, "--random-input-len")) if "--random-input-len" in command else None,
               output_len=int(option(command, "--hf-output-len") if "--hf-output-len" in command else option(command, "--random-output-len")),
               range_ratio=float(option(command, "--random-range-ratio")) if "--random-range-ratio" in command else None,
               prefix_caching=False, max_num_seqs=int(option(server, "--max-num-seqs")),
               max_num_batched_tokens=int(option(server, "--max-num-batched-tokens")),
               prometheus_scope="CLI invocation including initial request check" if run else "unavailable")
    before = path.with_name(path.stem + "-before.txt")
    after = path.with_name(path.stem + "-after.txt")
    row["prometheus"] = samples(before, after)
    row.update(diagnostics(folder, path))
    return row


def diagnostics(folder: Path, path: Path) -> dict:
    trace = folder / "av-trace.jsonl"
    window = path.with_name(path.stem + "-window.json")
    if not trace.is_file() or not window.is_file():
        return {}
    bounds = json.loads(window.read_text())
    records = trace_records(str(trace.resolve()))
    events = [r for r in records if bounds["start_ns"] <= r["time_ns"] <= bounds["end_ns"]]
    profiles = [r for r in records if r["kind"] == "costs" and r["time_ns"] <= bounds["start_ns"]]
    return dict(av_budget_steps=[r for r in events if r["kind"] == "budget"],
                av_output_steps=[r for r in events if r["kind"] == "output"],
                av_cost_profile=profiles[-1] if profiles else None,
                diagnostic_scope="CLI invocation including initial request check and warmups",
                instrumentation="av-diagnostics.patch, CPU-only JSONL logging")


@cache
def trace_records(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def smoke_rows(root: Path, metadata: dict) -> list[dict]:
    rows = []
    for path in sorted(root.glob("c[0-9]*.json")):
        data = json.loads(path.read_text())
        row = common_row(root, path, data, metadata)
        row.update(suite="initial-smoke", phase="smoke", av_enabled=True, repeat=None,
                   seed=None, execution_order={1: 1, 8: 2, 32: 3}[data["max_concurrency"]],
                   dataset="random", input_len=256, output_len=128, range_ratio=0.5,
                   temperature=None, prefix_caching=False, max_num_seqs=64,
                   max_num_batched_tokens=1024,
                   prometheus_scope="cumulative after-snapshot, not isolated run delta",
                   prometheus=samples(None, root / f"metrics-c{data['max_concurrency']}.txt"))
        rows.append(row)
    return rows


def common_row(root: Path, path: Path, data: dict, metadata: dict) -> dict:
    relative = str(path.relative_to(root))
    return dict(data, run_id=relative, source_json=relative,
                concurrency=data["max_concurrency"], target=metadata["target"],
                target_revision=metadata["target_revision"], draft=metadata["draft"],
                draft_revision=metadata["draft_revision"], k=metadata["num_speculative_tokens"],
                commit=metadata["commit"], gpu=metadata["gpu"],
                dtype=metadata["dtype"], model_runner=metadata["model_runner"],
                cuda_graph_mode=metadata["cuda_graph_mode"],
                profile_context_len=metadata["profile_context_len"],
                interactivity_tok_s_user=1000 / data["mean_tpot_ms"])


def samples(before: Path | None, after: Path) -> list[dict]:
    left, right = snapshot(before), snapshot(after)
    result = []
    for key in sorted(left.keys() | right.keys()):
        previous, current = left.get(key), right.get(key)
        result.append(dict(name=key[0], labels_json=key[1], before=previous, after=current,
                           delta=current - previous if previous is not None and current is not None else None))
    return result


def snapshot(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    families = text_string_to_metric_families(path.read_text())
    return {(s.name, json.dumps(s.labels, sort_keys=True)): float(s.value)
            for family in families for s in family.samples}


def option(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


if __name__ == "__main__":
    main()
