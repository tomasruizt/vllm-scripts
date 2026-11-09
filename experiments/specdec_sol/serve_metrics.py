"""Steady-window metrics of the `serve` pass runs, one long-format row per repeat and one per AV cost-table point.

Usage: serve_metrics.py 'results/gemma4-serve-*' results/gemma4-serve-analysis (run names are <model>-<kind>-<arm>-<date>)
"""

import json
import sys
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarking.aiperf import read_artifacts, server_stat, server_total, steady_window  # noqa: E402
from benchmarking.artifacts import write_parquet  # noqa: E402
from benchmarking.host_load import cell_span, peak_load, read_host_load  # noqa: E402

COST_ROWS = [256, 384, 448, 512, 513, 640, 768, 896, 1024, 1216, 1400]


def main() -> None:
    runs = [Path(path) for path in sorted(glob(sys.argv[1])) if Path(path).is_dir()]
    output = Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=True)
    write_parquet([row for run in runs for row in repeat_rows(run)], output / "metrics.parquet")
    write_parquet([row for run in runs for row in cost_rows(run)], output / "costs.parquet")


def repeat_rows(run: Path) -> list[dict]:
    summaries = sorted(glob(str(run / "c*/serve/aiperf-r*/profile_export_aiperf.json")))
    return [repeat_row(run, Path(summary)) for summary in summaries if error_rate(Path(summary)) == 0]


def error_rate(summary: Path) -> float:
    """Non-zero for a failed repeat, e.g. after an engine crash. AIPerf writes a plain 0.0 or a stat dict."""
    rate = json.loads(summary.read_text()).get("request_error_rate", 0.0)
    return rate["avg"] if isinstance(rate, dict) else rate


def repeat_row(run: Path, summary: Path) -> dict:
    path = summary.parent
    concurrency = int(path.parents[1].name[1:])
    artifacts = read_artifacts(summary)
    metrics = artifacts.server["metrics"]
    load = peak_load(read_host_load(run / "host-load.txt"), *cell_span(artifacts.requests))
    return {"arm": arm_name(run), "concurrency": concurrency, "repeat": int(path.name.removeprefix("aiperf-r")),
            **steady_window(artifacts.requests, concurrency), "acceptance_length": acceptance_length(metrics),
            "requests": len(artifacts.requests), "preemptions": server_total(metrics, "vllm:num_preemptions"),
            "running_p50": server_stat(metrics, "vllm:num_requests_running", "p50"),
            "waiting_p50": server_stat(metrics, "vllm:num_requests_waiting", "p50"), "host_load_peak": load}


def acceptance_length(metrics: dict) -> float | None:
    """Counts the bonus token. None for a target-only server."""
    if "vllm:spec_decode_num_drafts" not in metrics:
        return None
    return 1 + server_total(metrics, "vllm:spec_decode_num_accepted_tokens") / server_total(metrics, "vllm:spec_decode_num_drafts")


def cost_rows(run: Path) -> list[dict]:
    """AV's startup cost tables at a few batch sizes, for 127 requests. Fixed-budget arms and failed runs have none."""
    path = run / "c128/serve/av-costs.json"
    tables = json.loads(path.read_text())["results"][0] if path.exists() else None
    if tables is None:
        return []
    draft = tables["draft_ms"][127]
    return [{"arm": arm_name(run), "rows": rows, "draft_ms": draft, "verify_ms": tables["verify_ms"][rows],
             "cudagraph_limit": tables["cudagraph_limit"]} for rows in COST_ROWS]


def arm_name(run: Path) -> str:
    return run.name.split("-", 2)[2].rsplit("-", 1)[0]


if __name__ == "__main__":
    main()
