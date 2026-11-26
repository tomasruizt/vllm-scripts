"""Collect the fetched AgentX runs into results/metrics.parquet (long format) and print one table row per run.

Throughput follows the InferenceX convention: (input + output tokens per second) / GPUs. Interactivity is 1000 / mean ITL,
where AIPerf's ITL is per token, not per streamed chunk (a chunk carries several tokens under speculative decoding).
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarking.aiperf import server_stat, server_total
from benchmarking.artifacts import write_parquet

HERE = Path(__file__).parent
NUM_GPUS = 4
COLUMNS = ["concurrency", "host_offload", "throughput_per_gpu", "requests_per_s", "itl_ms", "interactivity", "ttft_ms",
           "cache_hit", "cache_hit_host", "input_tokens", "output_tokens", "kv_usage_peak", "running_mean", "requests", "errors"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Run ids under results/, e.g. full-c1-1")
    args = parser.parse_args()
    rows = sorted((summarize(HERE / "results" / run) for run in args.runs), key=lambda row: (row["concurrency"], row["run"]))
    long_rows = [{"run": row["run"], "metric": key, "value": float(row[key])} for row in rows for key in COLUMNS]
    write_parquet(long_rows, HERE / "results/metrics.parquet")
    print("| run | " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * (len(COLUMNS) + 1))
    for row in rows:
        print(f"| {row['run']} | " + " | ".join(cell(row[key]) for key in COLUMNS) + " |")


def cell(value: float) -> str:
    return f"{value:,.0f}" if value >= 100 or value == int(value) else f"{value:.3g}"


def summarize(root: Path) -> dict:
    client = json.loads((root / "aiperf_artifacts/profile_export_aiperf.json").read_text())
    server = json.loads((root / "aiperf_artifacts/server_metrics_export.json").read_text())["metrics"]
    tokens_per_s = client["input_token_throughput"]["avg"] + client["output_token_throughput"]["avg"]
    itl_ms = client["inter_token_latency"]["avg"]
    queried = server_total(server, "vllm:prefix_cache_queries")
    host_hits = server_total(server, "vllm:external_prefix_cache_hits")
    return {
        "run": root.name,
        "concurrency": int(re.search(r"-c(\d+)-", root.name).group(1)),  # run ids look like full-c16-1
        "host_offload": "SimpleCPUOffloadConnector" in (root / "server.log").read_text(errors="replace"),
        "throughput_per_gpu": tokens_per_s / NUM_GPUS,
        "requests_per_s": client["request_throughput"]["avg"],
        "itl_ms": itl_ms,
        "interactivity": 1000 / itl_ms,
        "ttft_ms": client["time_to_first_token"]["avg"],
        "cache_hit": (server_total(server, "vllm:prefix_cache_hits") + host_hits) / queried,
        "cache_hit_host": host_hits / queried,
        "input_tokens": client["input_sequence_length"]["avg"],
        "output_tokens": client["output_sequence_length"]["avg"],
        "kv_usage_peak": server_stat(server, "vllm:kv_cache_usage_perc", "max"),
        "running_mean": server_stat(server, "vllm:num_requests_running", "avg"),
        "requests": client["request_count"]["avg"],
        "errors": client.get("error_request_count", {}).get("avg", 0),
    }


if __name__ == "__main__":
    main()
