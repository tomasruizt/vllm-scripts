"""Audit retained AL counters and identify runs with matching prompt sequences."""

import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarking.aiperf import server_total
from benchmarking.artifacts import read_jsonl, write_csv, write_parquet

HERE = Path(__file__).resolve().parent


def main() -> None:
    source = HERE / "results/dspark-markov-56694-20261102-r2"
    with (source / "metrics.csv").open() as stream:
        rows = [audit(row) for row in csv.DictReader(stream) if row["included"].lower() == "true"]
    assert len(rows) == 48
    write_parquet(rows, source / "acceptance-audit.parquet")
    write_csv(rows, source / "acceptance-audit.csv")
    print(json.dumps(dict(runs=len(rows), positions=sum(len(r["accepted_per_position"]) for r in rows),
                          max_al_error=max(r["al_error"] for r in rows),
                          generation_matches=sum(r["generation_tokens"] == r["client_output_tokens"] for r in rows),
                          fixed_k8=sum(r["draft_tokens"] == 8 * r["drafts"] for r in rows)), indent=2))


def audit(row: dict) -> dict:
    path = HERE / "results" / row["suite"] / row["source_json"]
    data = json.loads(path.with_name("server_metrics_export.json").read_text())
    metrics = data["metrics"]
    drafts = server_total(metrics, "vllm:spec_decode_num_drafts")
    accepted = server_total(metrics, "vllm:spec_decode_num_accepted_tokens")
    draft_tokens = server_total(metrics, "vllm:spec_decode_num_draft_tokens")
    series = metrics["vllm:spec_decode_num_accepted_tokens_per_pos"]["series"]
    positions = [sum(s["stats"]["total"] for s in series if int(s["labels"]["position"]) == i) for i in range(8)]
    assert len(series) == 8 and drafts > 0
    assert {int(s["labels"]["position"]) for s in series} == set(range(8))
    assert draft_tokens == 8 * drafts
    assert sum(positions) == accepted
    assert all(0 <= n <= drafts for n in positions)
    assert all(a >= b for a, b in zip(positions, positions[1:]))
    al = 1 + accepted / drafts
    error = abs(al - float(row["acceptance_length"]))
    assert error < 1e-12
    phase = data["summary"]["phase_time_ranges"]["profiling"]
    endpoints = data["summary"]["endpoint_info"].values()
    assert data["metrics_phase"] == "profiling"
    assert all(e["first_fetch_ns"] <= phase["start_ns"] and e["last_fetch_ns"] >= phase["end_ns"] for e in endpoints)
    records = sorted(read_jsonl(path.with_name("profile_export.jsonl")), key=lambda r: r["metadata"]["request_start_ns"])
    ids = [r["metadata"]["conversation_id"] for r in records]
    output_lengths = [r["metrics"]["output_sequence_length"]["value"] for r in records]
    generation = server_total(metrics, "vllm:generation_tokens")
    assert generation == sum(output_lengths)
    return dict(suite=row["suite"], mode=row["mode"], workload=row["workload"],
                concurrency=int(row["concurrency"]), trial=int(row["trial"]), requests=len(records),
                drafts=drafts, draft_tokens=draft_tokens, accepted=accepted,
                accepted_per_position=positions, acceptance_length=al, al_error=error,
                generation_tokens=generation, client_output_tokens=sum(output_lengths),
                prompt_sequence=signature(ids), output_lengths=signature(output_lengths))


def signature(values: list) -> str:
    return sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


if __name__ == "__main__":
    main()
