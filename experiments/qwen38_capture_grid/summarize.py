"""Collect native AIPerf results and graph-dispatch statistics locally."""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen_dflash_av_60229"))
from build_report import server_stat, server_total, steady_window

LABELS = {"default": "Default", "2k": "Up to 2K", "4k": "Up to 4K"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[32, 128])
    args = parser.parse_args()
    recipe = args.root.name.startswith("qwen38-recipe")
    rows = [load_cell(path, args.root) for path in sorted(args.root.glob("*/artifacts/**/profile_export_aiperf.json"))]
    repeats = [1, 2, 3] if recipe else [1]
    expected = {(label, concurrency, repeat) for label in LABELS.values() for concurrency in args.concurrencies for repeat in repeats}
    actual = {(row["variant"], row["concurrency"], row["repeat"]) for row in rows}
    assert args.partial or (len(rows) == len(expected) and actual == expected), f"Incomplete experiment: {actual}"
    (args.root / "metrics.json").write_text(json.dumps(rows) + "\n")
    keys = ["variant", "concurrency", "valid", "requests", "window_s", "throughput", "mean_tpot_ms", "mean_ttft_ms", "startup_s", "graph_memory_gib", "kv_tokens"]
    for row in rows:
        print(json.dumps({key: row[key] for key in keys}))


def load_cell(path: Path, root: Path) -> dict:
    variant = path.relative_to(root).parts[0]
    recipe = root.name.startswith("qwen38-recipe")
    repeat_match = re.search(r"/r(\d+)-c\d+/", str(path))
    directory = root / variant
    data = json.loads(path.read_text())
    server = json.loads(path.with_name("server_metrics_export.json").read_text())
    records = [json.loads(line) for line in path.with_name("profile_export.jsonl").read_text().splitlines()]
    done = [record for record in records if "output_sequence_length" in record.get("metrics", {})]
    phase = next(phase for phase in data["input_config"]["phases"] if phase["name"] == "profiling")
    concurrency = phase["concurrency"]
    steady = steady_window(done, concurrency)
    metrics = server["metrics"]
    cancelled = sum(record["metadata"].get("was_cancelled", False) for record in records)
    errors = data["request_error_rate"]["avg"]
    preemptions = server_total(metrics, "vllm:num_preemptions")
    running = server_stat(metrics, "vllm:num_requests_running", "p50")
    waiting = server_stat(metrics, "vllm:num_requests_waiting", "p50")
    checks = dict(all_completed=len(done) == len(records) and cancelled == 0, no_errors=errors == 0,
                  no_preemptions=preemptions == 0, saturated=running >= 0.9 * concurrency,
                  no_queueing=waiting == 0, window_ge_10s=steady["window_s"] >= 10)
    log = (directory / "server.log").read_text()
    row = dict(suite=root.name, variant=LABELS[variant], repeat=int(repeat_match[1]) if repeat_match else 1,
               concurrency=concurrency, requests=len(done),
               submission_duration_s=phase["duration"], grace_period_s=phase["grace_period"],
               model="Qwen/Qwen3.8-Flash-Next", model_revision="de4b8e4d43b917e7706784d8bb445c9af86a3540",
               weight_dtype="bfloat16", kv_cache_dtype="fp8", tensor_parallel_size=4, expert_parallel=True,
               num_speculative_tokens=3, av_enabled=False, prefix_caching=False,
               max_num_seqs=256, max_num_batched_tokens=8192, max_model_len=16384, embedding_cpu_offload=True,
               capture_cap={"default": 1024, "2k": 2048, "4k": 4096}[variant],
               valid=all(checks.values()), failed_gates=[name for name, ok in checks.items() if not ok],
               cancelled=cancelled, error_rate=errors, preemptions=preemptions,
               server_running_p50=running, server_waiting_p50=waiting,
               mean_input_tokens=data["input_sequence_length"]["avg"], mean_output_tokens=data["output_sequence_length"]["avg"],
               aiperf_throughput=data["output_token_throughput"]["avg"],
               startup_s=elapsed(directory / "start.txt", directory / "ready.txt"),
               graph_memory_gib=last_number(log, r"Graph capturing finished in .*?took ([\d.]+) GiB"),
               kv_tokens=last_number(log, r"GPU KV cache size: ([\d,]+) tokens"),
               graph_capture_s=last_number(log, r"Graph capturing finished in ([\d.]+) secs"),
               runtime_commit=(root / "runtime-commit.txt").read_text().strip(),
               server_command=(directory / "server-command.txt").read_text().strip(),
               aiperf_summary_json=json.dumps(data), aiperf_server_metrics_json=json.dumps(server),
               aiperf_request_records_json=json.dumps(records), **steady)
    row.update(graph_stats(log, row["capture_cap"]))
    if recipe:
        info = json.loads((directory / "server-info.json").read_text())
        config = info["vllm_config"]
        row.update(model="Qwen/Qwen3.8-Flash-Next-FP8", model_revision="236dfdf285828023ca3bcd3f37366c58a3469b13",
                   weight_dtype="fp8", prefix_caching=True, max_model_len=config["model_config"]["max_model_len"],
                   max_num_batched_tokens=config["scheduler_config"]["max_num_batched_tokens"], server_info_json=json.dumps(info))
    return row


def elapsed(start: Path, end: Path) -> float:
    return (datetime.fromisoformat(end.read_text().strip()) - datetime.fromisoformat(start.read_text().strip())).total_seconds()


def last_number(log: str, pattern: str) -> float | None:
    values = re.findall(pattern, log)
    return float(values[-1].replace(",", "")) if values else None


def graph_stats(log: str, cap: int) -> dict:
    counts = Counter()
    shapes = Counter()
    padding = 0
    tokens = 0
    pattern = r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\w+)\s*\|\s*(\d+)\s*\|"
    for match in re.finditer(pattern, log):
        count = int(match[5])
        counts[match[4]] += count
        shapes[(int(match[1]), int(match[2]), match[4])] += count
        padding += int(match[3]) * count
        tokens += int(match[1]) * count
    histogram = [dict(tokens=key[0], padded_tokens=key[1], mode=key[2], count=count) for key, count in sorted(shapes.items())]
    eager_above = sum(shape["count"] for shape in histogram if shape["mode"] == "NONE" and shape["tokens"] > cap)
    return dict(session_graph_mode_counts=dict(counts), session_graph_shapes=histogram,
                session_eager_above_cap_steps=eager_above, session_padding_ratio=padding / tokens if tokens else None)


if __name__ == "__main__":
    main()
