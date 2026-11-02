"""Experiment configuration for the shared AIPerf report builder."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarking.aiperf import gates, server_stat, server_total, steady_window
from benchmarking.artifacts import write_csv
from reporting.aiperf_report import ReportConfig, load_run as load_report_run, main as build_report, plt


PUBLIC_PATHS = {
    "/home/tomasruizt/code/vllm-dsv41/.venv/bin/python": "python",
    "/mnt/nfs-preprod-1/engine/hub_cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218": "Qwen/Qwen3-8B",
    "/data/tomasruizt/benchmarks/speed-bench/throughput_1k.jsonl": "speed_bench_throughput_1k.jsonl",
}

CONFIG = ReportConfig(
    title="Qwen3-8B XPress + Adaptive Verification",
    subtitle="8B BF16 · one H100 per server, TP=1 · XPress b16, k=15 · SPEED-Bench throughput-1K · AIPerf 0.13.0",
    download_prefix="qwen3-8b-xpress", draft_tokens=15,
    summary_fields=("suite", "commit"),
    public_paths=tuple(PUBLIC_PATHS.items()),
)


def load_run(path: Path, root: Path) -> dict:
    """Compatibility entry point for this experiment's existing analyses."""
    return load_report_run(path, root, instrumentation=CONFIG.instrumentation)


if __name__ == "__main__":
    build_report(CONFIG)
