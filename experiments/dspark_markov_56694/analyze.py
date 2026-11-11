"""Collect DSpark pruning measurements using the shared AIPerf metric definitions."""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarking.aiperf import gates, read_artifacts, server_stat, server_total, steady_window
from benchmarking.host_load import cell_span, peak_load, read_host_load
from benchmarking.artifacts import write_csv, write_parquet

EXCLUDED_SUITES = {"dspark-markov-56694-20261102-c4": "gpu5_91C_345MHz"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--extra-source", type=Path, action="append", default=[])
    parser.add_argument("--max-host-load", type=float, default=None,
                        help="reject cells whose peak busy CPU cores exceed this value")
    parser.add_argument("--self-saturating", action="store_true",
                        help="fixed-work closed-loop cells drain by design, skip the saturation gate")
    args = parser.parse_args()
    sources = [args.source, *args.extra_source]
    rows = [collect(path, source, args.self_saturating, args.max_host_load) for source in sources
            for path in sorted(source.rglob("profile_export_aiperf.json"))]
    if rows:
        write_parquet(rows, args.source / "metrics.parquet")
        write_csv(rows, args.source / "metrics.csv")
    agreement = {source.name: output_agreement(source) for source in sources}
    dense = [json.loads(path.read_text()) for source in sources if source.name not in EXCLUDED_SUITES
             for path in sorted(source.glob("dense_r*/outputs.json")) or sorted(source.glob("dense/outputs.json"))]
    dense_repeat = [matching_outputs(dense[0], values) for values in dense[1:]]
    audit = dict(cells=len(rows), valid=sum(row["valid"] for row in rows),
                 included=sum(row["included"] for row in rows),
                 greedy_agreement=agreement, dense_startup_agreement=dense_repeat)
    (args.source / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


def output_agreement(source: Path) -> dict:
    outputs = {path.parent.name: json.loads(path.read_text()) for path in source.glob("*/outputs.json")}
    reference = next((values for name, values in outputs.items() if name.startswith("dense")), None)
    if reference is None:
        return {}
    return {name: matching_outputs(reference, values) for name, values in outputs.items()}


def matching_outputs(left: list[dict], right: list[dict]) -> int:
    return sum(a["choices"][0]["message"] == b["choices"][0]["message"]
               for a, b in zip(left, right, strict=True))


def mode_host_load(path: Path, requests: list[dict]) -> float | None:
    """Peak busy cores over the cell, from the log written next to this mode's server."""
    log = next((parent / "host-load.txt" for parent in path.parents
                if (parent / "host-load.txt").exists()), None)
    return peak_load(read_host_load(log), *cell_span(requests)) if log else None


def split_mode(name: str) -> tuple[str, int]:
    """Recover the mode and round from a rotated `mode_rN` directory name."""
    match = re.fullmatch(r"(.*)_r(\d+)", name)
    return (match.group(1), int(match.group(2))) if match else (name, 0)


def collect(path: Path, source: Path, self_saturating: bool = False,
            max_host_load: float | None = None) -> dict:
    artifacts = read_artifacts(path, phase_name="profiling")
    summary = artifacts.summary
    metrics = artifacts.server["metrics"]
    requests = artifacts.requests
    done = artifacts.completed
    config = next(p for p in summary["input_config"]["phases"] if p["name"] == "profiling")
    parts = path.relative_to(source).parts
    mode, round_index = split_mode(parts[0])
    trial = next((part for part in parts if part.startswith(("trial_", "run_"))), "trial_0")
    drafts = server_total(metrics, "vllm:spec_decode_num_drafts")
    row = dict(suite=source.name, mode=mode, round=round_index, workload=parts[1],
               trial=int(trial.split("_")[-1]),
               source_json=str(path.relative_to(source)),
               concurrency=config["concurrency"], requests_target=config.get("requests") or len(requests),
               requests=len(done), cancelled=sum(r["metadata"].get("was_cancelled", False) for r in requests),
               error_rate=summary["request_error_rate"]["avg"],
               preemptions=server_total(metrics, "vllm:num_preemptions"),
               server_running_p50=server_stat(metrics, "vllm:num_requests_running", "p50"),
               server_waiting_p50=server_stat(metrics, "vllm:num_requests_waiting", "p50"),
               acceptance_length=1 + server_total(metrics, "vllm:spec_decode_num_accepted_tokens") / drafts if drafts else None,
               aiperf_throughput=summary["output_token_throughput"]["avg"],
               aiperf_tpot_ms=summary["inter_token_latency"]["avg"],
               mean_input_tokens=summary["input_sequence_length"]["avg"],
               mean_output_tokens=summary["output_sequence_length"]["avg"],
               **steady_window(done, config["concurrency"]))
    row["host_load_peak"] = mode_host_load(path, requests)
    row.update(gates(row, min_window_s=10, require_saturation=not self_saturating,
                     max_host_load=max_host_load))
    row["hardware_valid"] = source.name not in EXCLUDED_SUITES
    row["hardware_exclusion"] = EXCLUDED_SUITES.get(source.name, "")
    row["valid"] = row["valid"] and row["hardware_valid"]
    row["included"] = row["valid"] and row["concurrency"] in (1, 4, 8, 32)
    return row


if __name__ == "__main__":
    main()
