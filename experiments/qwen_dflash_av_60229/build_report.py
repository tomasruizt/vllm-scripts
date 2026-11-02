"""Experiment configuration for the shared AIPerf report builder."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarking.aiperf import gates, server_stat, server_total, steady_window
from benchmarking.artifacts import write_csv
from reporting.aiperf_report import ReportConfig, load_run as load_report_run, main as build_report, plt


PUBLIC_PATHS = {
    "/home/tomasruizt/.venv/bin/python -m vllm.entrypoints.openai.api_server --model": "vllm serve",
    "/data/engine/hub_cache/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754": "Qwen/Qwen3.5-35B-A3B",
    "/data/tomasruizt/hub_cache/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754": "Qwen/Qwen3.5-35B-A3B",
    "/data/tomasruizt/benchmarks/deepseek-v4-dep4-av/dataset/throughput_1k.jsonl": "speed_bench_throughput_1k.jsonl",
}

CONFIG = ReportConfig(
    title="Qwen3.5 + Adaptive Verification",
    subtitle="35B-A3B BF16 · one B300 per server, TP=1 · DFlash k=7 · SPEED-Bench throughput-1K · AIPerf 0.13.0",
    download_prefix="qwen35-av", draft_tokens=7,
    instrumentation="none, unchanged PR60229",
    default_capture_note="Default CUDA graph capture sizes (up to 1,024 tokens on B300): steps with a prefill exceed them and run eagerly.",
    public_paths=tuple(PUBLIC_PATHS.items()),
)


def load_run(path: Path, root: Path) -> dict:
    """Compatibility entry point for this experiment's existing analyses."""
    return load_report_run(path, root, instrumentation=CONFIG.instrumentation)


if __name__ == "__main__":
    build_report(CONFIG)
