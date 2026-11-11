"""Collect the serving A/B with shared AIPerf metric definitions and gates."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from benchmarking.aiperf import gates, read_artifacts, server_stat, server_total, steady_window
from benchmarking.artifacts import write_csv, write_parquet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    source = args.run / "results"
    cells = json.loads((source / "cells.json").read_text())
    rows = []
    requests = []
    for cell in cells:
        row, records = collect(source, cell)
        rows.append(row)
        requests.extend(records)
    output = args.run / "analysis"
    output.mkdir(exist_ok=True)
    write_csv(rows, output / "serving.csv")
    write_parquet(rows, output / "serving.parquet")
    write_parquet(requests, output / "serving-requests.parquet")
    summary = dict(cells=len(rows), valid=sum(row["valid"] for row in rows),
                   requests=sum(row["requests"] for row in rows),
                   errors=sum(row["errors"] for row in rows),
                   cancelled=sum(row["cancelled"] for row in rows))
    print(json.dumps(summary))
    if len(rows) != 12 or not all(row["valid"] for row in rows):
        raise SystemExit(2)


def collect(source: Path, cell: dict) -> tuple[dict, list[dict]]:
    path = source / cell["directory"] / "aiperf-measurement/profile_export_aiperf.json"
    artifacts = read_artifacts(path)
    summary = artifacts.summary
    metrics = artifacts.server["metrics"]
    phase = summary["input_config"]["phases"][0]
    drafts = server_total(metrics, "vllm:spec_decode_num_drafts")
    accepted = server_total(metrics, "vllm:spec_decode_num_accepted_tokens")
    prompts = sorted(row["metadata"]["conversation_id"] for row in artifacts.requests)
    row = dict(**cell, run_id=source.parent.name, phase_duration_s=phase["duration"],
               requests=len(artifacts.completed), requests_target=phase.get("requests"),
               errors=len(artifacts.requests) - len(artifacts.completed),
               error_rate=summary["request_error_rate"]["avg"],
               cancelled=sum(item["metadata"].get("was_cancelled", False) for item in artifacts.requests),
               total_output_tokens=summary["total_osl"]["avg"],
               mean_input_tokens=summary["input_sequence_length"]["avg"],
               mean_output_tokens=summary["output_sequence_length"]["avg"],
               whole_tpot_ms=summary["inter_token_latency"]["avg"],
               whole_ttft_ms=summary["time_to_first_token"]["avg"],
               whole_throughput=summary["output_token_throughput"]["avg"],
               acceptance_length=1 + accepted / drafts, accepted_tokens=accepted, draft_steps=drafts,
               server_running_p50=server_stat(metrics, "vllm:num_requests_running", "p50"),
               server_waiting_p50=server_stat(metrics, "vllm:num_requests_waiting", "p50"),
               preemptions=server_total(metrics, "vllm:num_preemptions"),
               prompt_set=sha256("\n".join(prompts).encode()).hexdigest()[:12],
               **steady_window(artifacts.completed, cell["concurrency"]))
    row.update(gates(row, min_window_s=15))
    records = [request_metrics(cell, item) for item in artifacts.completed]
    assert sum(item["output_tokens"] for item in records) == row["total_output_tokens"]
    return row, records


def request_metrics(cell: dict, request: dict) -> dict:
    metadata = request["metadata"]
    metrics = request["metrics"]
    return dict(repeat=cell["repeat"], mode=cell["mode"], concurrency=cell["concurrency"],
                conversation_id=metadata["conversation_id"], turn_index=metadata["turn_index"],
                start_ns=metadata["request_start_ns"], end_ns=metadata["request_end_ns"],
                input_tokens=metrics["input_sequence_length"]["value"],
                output_tokens=metrics["output_sequence_length"]["value"],
                tpot_ms=metrics.get("inter_token_latency", {}).get("value"),
                ttft_ms=metrics["time_to_first_token"]["value"])


if __name__ == "__main__":
    main()
