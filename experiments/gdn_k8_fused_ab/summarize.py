"""Collect AIPerf cells of the GDN K8 fused A/B into one long-format parquet."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarking.aiperf import gates, read_artifacts, server_stat, server_total, steady_window


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = [load_cell(path, args.root) for path in sorted(args.root.glob("*/artifacts/*/profile_export_aiperf.json"))]
    output = args.root / "metrics.json"
    output.write_text(json.dumps(rows) + "\n")
    duckdb.execute(f"COPY (SELECT * FROM read_json_auto('{output}')) TO '{args.root / 'metrics.parquet'}' (FORMAT PARQUET)")
    print(f"{len(rows)} cells -> {args.root / 'metrics.parquet'}")


def load_cell(path: Path, root: Path) -> dict:
    variant, _, cell = path.relative_to(root).parts[:3]
    repeat, concurrency = map(int, re.fullmatch(r"r(\d+)-c(\d+)", cell).groups())
    artifacts = read_artifacts(path)
    metrics = artifacts.server["metrics"]
    accepted = server_total(metrics, "vllm:spec_decode_num_accepted_tokens")
    drafts = server_total(metrics, "vllm:spec_decode_num_drafts")
    cancelled = sum(r["metadata"].get("was_cancelled", False) for r in artifacts.requests)
    row = dict(variant=variant, repeat=repeat, concurrency=concurrency, requests=len(artifacts.completed),
               requests_target=None, cancelled=cancelled, error_rate=artifacts.summary["request_error_rate"]["avg"],
               preemptions=server_total(metrics, "vllm:num_preemptions"),
               server_running_p50=server_stat(metrics, "vllm:num_requests_running", "p50"),
               server_waiting_p50=server_stat(metrics, "vllm:num_requests_waiting", "p50"),
               acceptance_length=1 + accepted / drafts if drafts else None,
               aiperf_output_tps=artifacts.summary["output_token_throughput"]["avg"],
               commit=(root / variant / "commit.txt").read_text().strip(),
               **steady_window(artifacts.completed, concurrency))
    return row | gates(row) | graph_modes((root / variant / "server.log").read_text()) | session_info(root / variant)


def session_info(variant_dir: Path) -> dict:
    """GPU of the server session and its GSM8K accuracy (5-shot, greedy), when present."""
    gsm8k = variant_dir / "gsm8k.json"
    gpu = variant_dir / "gpu.txt"
    return dict(gsm8k_accuracy=json.loads(gsm8k.read_text())["accuracy"] if gsm8k.exists() else None,
                gpu=gpu.read_text().strip() if gpu.exists() else None)


def graph_modes(log: str) -> dict:
    """Step counts per CUDA graph mode from the --cudagraph-metrics tables over the whole server session."""
    counts = Counter()
    for match in re.finditer(r"\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\w+)\s*\|\s*(\d+)\s*\|", log):
        counts[match[1]] += int(match[2])
    return dict(session_graph_mode_counts=json.dumps(counts))


if __name__ == "__main__":
    main()
