"""Gather a run into `<run>/metrics.parquet` (long format) and print the variant comparison."""

import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

# metric -> (export file, key, scale). Server histograms and client metrics, averages.
SERVER = {"prefill_ms": "vllm:request_prefill_time_seconds", "decode_ms": "vllm:request_decode_time_seconds",
          "server_ttft_ms": "vllm:time_to_first_token_seconds", "queue_ms": "vllm:request_queue_time_seconds"}
CLIENT = {"ttft_ms": "time_to_first_token", "itl_ms": "inter_token_latency", "output_tps": "output_token_throughput",
          "requests": "request_count"}
COUNTERS = {"drafts": "vllm:spec_decode_num_drafts", "draft_tokens": "vllm:spec_decode_num_draft_tokens",
            "verified_tokens": "vllm:spec_decode_num_verified_tokens",
            "accepted_tokens": "vllm:spec_decode_num_accepted_tokens"}


def main(run_dir: Path) -> None:
    cells = sorted(run_dir.glob("gpu*/*/*-trial-*/artifacts"))
    metrics = pd.DataFrame([row for cell in cells for row in cell_rows(cell)])
    metrics.to_parquet(run_dir / "metrics.parquet")
    print(comparison(run_dir / "metrics.parquet").to_string(index=False))
    print(lazy_vs_main(run_dir / "metrics.parquet").to_string(index=False))


def cell_rows(artifacts: Path) -> list[dict]:
    gpu, variant, cell = artifacts.parts[-4:-1]
    workload, trial = cell.split("-trial-")
    client = json.loads((artifacts / "profile_export_aiperf.json").read_text())
    server = json.loads((artifacts / "server_metrics_export.json").read_text())["metrics"]
    values = {name: client[key]["avg"] for name, key in CLIENT.items()}
    values |= {f"ttft_{q}_ms": client["time_to_first_token"][q] for q in ("p50", "p99")}
    values |= {name: 1e3 * series_stats(server, key)["avg"] for name, key in SERVER.items()}
    values |= {name: counter_delta(server, key) for name, key in COUNTERS.items()}
    base = dict(gpu=gpu, variant=variant, workload=workload, trial=int(trial))
    return [dict(base, metric=name, value=float(value)) for name, value in values.items()]


def series_stats(server: dict, key: str) -> dict:
    return server[key]["series"][0]["stats"]


def counter_delta(server: dict, key: str) -> float:
    stats = series_stats(server, key) if key in server else {}
    return next((stats[name] for name in ("total", "delta", "sum") if name in stats), float("nan"))


def comparison(parquet: Path) -> pd.DataFrame:
    """Mean over GPUs and trials, and the change against fixed K (`k5`) and main's AV (`av_main`)."""
    return duckdb.sql(f"""
        with cells as (pivot '{parquet}' on metric using first(value)),
        v as (select workload, variant, count(*) cells, avg(prefill_ms) prefill_ms, min(prefill_ms) prefill_min,
                     max(prefill_ms) prefill_max, avg(server_ttft_ms) server_ttft_ms, avg(decode_ms) decode_ms,
                     avg(itl_ms) itl_ms, avg(output_tps) output_tps,
                     1 + sum(accepted_tokens) / sum(drafts) al, sum(verified_tokens) / sum(draft_tokens) verified
              from cells group by all)
        select workload, variant, cells, round(prefill_ms, 1) prefill_ms,
               round(prefill_min, 1) || '-' || round(prefill_max, 1) prefill_range,
               round(prefill_ms - max(prefill_ms) filter (variant = 'k5') over w, 1) prefill_vs_k5,
               round(server_ttft_ms, 1) server_ttft_ms, round(decode_ms, 1) decode_ms, round(itl_ms, 2) itl_ms,
               round(output_tps) output_tps,
               round(100 * (output_tps / max(output_tps) filter (variant = 'av_main') over w - 1), 1) tps_vs_av_main_pct,
               round(al, 2) al, round(100 * verified, 1) verified_pct
        from v window w as (partition by workload)
        order by workload desc, variant""").df()


def lazy_vs_main(parquet: Path) -> pd.DataFrame:
    """Per GPU and workload: vllm#63355 minus main, in ms (throughput in %)."""
    return duckdb.sql(f"""
        with cells as (pivot '{parquet}' on metric using first(value)),
        g as (select workload, gpu, variant, avg(prefill_ms) prefill, avg(server_ttft_ms) server_ttft, avg(queue_ms) queue,
                     avg(ttft_ms) ttft, avg(ttft_p50_ms) ttft_p50, avg(ttft_p99_ms) ttft_p99, avg(output_tps) tps
              from cells where variant in ('av_main', 'av_lazy') group by all)
        select workload, gpu, round(m.server_ttft, 1) main_server_ttft_ms, round(l.prefill - m.prefill, 2) d_prefill,
               round(l.queue - m.queue, 2) d_queue, round(l.server_ttft - m.server_ttft, 2) d_server_ttft,
               round(l.ttft - m.ttft, 2) d_client_ttft, round(l.ttft_p50 - m.ttft_p50, 2) d_client_p50,
               round(l.ttft_p99 - m.ttft_p99, 2) d_client_p99, round(100 * (l.tps / m.tps - 1), 2) d_tps_pct
        from g m join g l using (workload, gpu) where m.variant = 'av_main' and l.variant = 'av_lazy'
        order by workload, gpu""").df()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
