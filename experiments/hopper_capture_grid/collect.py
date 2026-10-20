"""Collect original AIPerf metrics and startup costs in one long-format dataset."""
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen_dflash_av_60229"))
from build_report import server_stat, server_total, steady_window


def main() -> None:
    root = Path(sys.argv[1])
    rows = [load_cell(path, root) for path in sorted(root.glob("*/*/startup-*/c*/artifacts/**/profile_export_aiperf.json"))]
    startups = [startup_row(path) for path in sorted(root.glob("*/*/startup-*/server.log"))]
    for case in ("dense", "gdn"):
        for variant in ("default", "4k"):
            directory = root / case / variant
            code = int((directory / "exit-code.txt").read_text())
            cells = [row for row in rows if row["case"] == case and row["variant"] == variant]
            assert code != 0 or len(cells) == 18, f"Incomplete successful worker: {directory}"
            assert code == 0 or list(directory.glob("startup-*/failure.txt")), directory
    (root / "metrics.json").write_text(json.dumps(rows) + "\n")
    (root / "startup-metrics.json").write_text(json.dumps(startups) + "\n")
    print(f"Collected {len(rows)} cells, {sum(row['requests'] for row in rows)} completed requests")


def load_cell(path: Path, root: Path) -> dict:
    case, variant, startup, cell = path.relative_to(root).parts[:4]
    directory = root / case / variant / startup
    data = json.loads(path.read_text())
    server = json.loads(path.with_name("server_metrics_export.json").read_text())
    records = [json.loads(line) for line in path.with_name("profile_export.jsonl").read_text().splitlines()]
    done = [record for record in records if "output_sequence_length" in record.get("metrics", {})]
    concurrency = int(cell.removeprefix("c"))
    metrics = server["metrics"]
    log = (directory / "server.log").read_text()
    info = json.loads((directory / "server-info.json").read_text())
    start = int((directory / "start-ns.txt").read_text())
    ready = int((directory / "ready-ns.txt").read_text())
    steady = steady_window(done, concurrency)
    cancelled = sum(record["metadata"].get("was_cancelled", False) for record in records)
    checks = dict(all_completed=len(done) == len(records) and cancelled == 0,
                  no_errors=data["request_error_rate"]["avg"] == 0,
                  no_preemptions=server_total(metrics, "vllm:num_preemptions") == 0,
                  saturated=server_stat(metrics, "vllm:num_requests_running", "p50") >= 0.9 * concurrency,
                  no_queueing=server_stat(metrics, "vllm:num_requests_waiting", "p50") == 0,
                  window_ge_10s=steady["window_s"] >= 10)
    return dict(case=case, variant=variant, startup=int(startup.removeprefix("startup-")), concurrency=concurrency,
                requests=len(done), records=len(records), error_rate=data["request_error_rate"]["avg"],
                cancelled=cancelled, valid=all(checks.values()), failed_gates=[key for key, value in checks.items() if not value],
                preemptions=server_total(metrics, "vllm:num_preemptions"),
                running_p50=server_stat(metrics, "vllm:num_requests_running", "p50"),
                waiting_p50=server_stat(metrics, "vllm:num_requests_waiting", "p50"),
                startup_s=(ready-start)/1e9,
                graph_memory_gib=number(log, r"CUDA graph pool memory: ([\d.]+) GiB \(actual\)"),
                estimated_graph_memory_gib=number(log, r"GiB \(actual\), ([\d.]+) GiB \(estimated\)"),
                kv_memory_gib=number(log, r"Available KV cache memory: ([\d.]+) GiB"),
                kv_tokens=number(log, r"GPU KV cache size: ([\d,]+) tokens"),
                capture_s=number(log, r"Graph capturing finished in ([\d.]+) secs"),
                server_command_json=(directory / "server-command.json").read_text(),
                server_info_json=json.dumps(info), aiperf_summary_json=json.dumps(data),
                aiperf_server_metrics_json=json.dumps(server), aiperf_request_records_json=json.dumps(records),
                **steady)


def startup_row(path: Path) -> dict:
    directory = path.parent
    log = path.read_text()
    ready = directory / "ready-ns.txt"
    failure = directory / "failure.txt"
    return dict(case=directory.parents[1].name, variant=directory.parent.name,
                startup=int(directory.name.removeprefix("startup-")),
                startup_s=(int(ready.read_text())-int((directory / "start-ns.txt").read_text()))/1e9 if ready.exists() else None,
                graph_memory_gib=number(log, r"CUDA graph pool memory: ([\d.]+) GiB \(actual\)"),
                estimated_graph_memory_gib=number(log, r"GiB \(actual\), ([\d.]+) GiB \(estimated\)"),
                kv_memory_gib=number(log, r"Available KV cache memory: ([\d.]+) GiB"),
                kv_tokens=number(log, r"GPU KV cache size: ([\d,]+) tokens"),
                capture_s=number(log, r"Graph capturing finished in ([\d.]+) secs"),
                failure=failure.read_text() if failure.exists() else None)


def number(text: str, pattern: str) -> float | None:
    matches = re.findall(pattern, text)
    return float(matches[-1].replace(",", "")) if matches else None


if __name__ == "__main__":
    main()
