"""Gate a native AIPerf concurrency sweep, then run five more trials per policy."""

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from urllib.request import Request, urlopen

import yaml

import build_report


CONCURRENCIES = [1, 8, 32, 64, 128, 256]
CURVE_SHA256 = "43a774f8e1674dde929b6a6e45f03ad1fc75adb016f56f87774f5252954a9368"


def main() -> None:
    root = Path(sys.argv[1])
    curve = root / "measured-curves-frozen.json"
    if hashlib.sha256(curve.read_bytes()).hexdigest() != CURVE_SHA256:
        raise RuntimeError("Wrong frozen curve")
    durations = dict.fromkeys(CONCURRENCIES, 30)
    rows = []
    for mode in ("native", "measured"):
        rows += run_sweep(root, mode, f"pilot-{mode}", durations, 1)
    reject_errors(rows)
    for c in CONCURRENCIES:
        window = min(row["window_s"] for row in rows if row["concurrency"] == c)
        if window < 20:
            durations[c] = math.ceil(30 + 25 - window)
    extended = {c: duration for c, duration in durations.items() if duration > 30}
    if extended:
        for row in rows:
            if row["concurrency"] in extended:
                row["included"] = False
        for mode in ("measured", "native"):
            rows += run_sweep(root, mode, f"pilot-extended-{mode}", extended, 1)
    validate_rows([row for row in rows if row["included"]])
    (root / "pilot-validation.json").write_text(json.dumps(rows, indent=2))
    print(f"PILOT_VALIDATED durations={durations}", flush=True)
    for mode in ("measured", "native"):
        rows += run_sweep(root, mode, f"main-{mode}", durations, 5)
    validate_rows([row for row in rows if row["included"]])
    for mode in ("native", "measured"):
        for c in CONCURRENCIES:
            count = sum(row["included"] and row["mode"] == mode and row["concurrency"] == c for row in rows)
            if count != 6:
                raise RuntimeError(f"Expected six valid runs for {mode} c{c}, got {count}")
    (root / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "DONE").write_text("72 matched TP2 sweep runs passed serving and coverage gates.\n")
    print("FULL_SWEEP_DONE", flush=True)


def run_sweep(root: Path, mode: str, stage: str, durations: dict[int, int], repeats: int) -> list[dict]:
    directory = root / stage
    directory.mkdir()
    config = yaml.safe_load(Path(__file__).with_name("full-curve-sweep.yaml").read_text())
    config["sweep"]["parameters"] = dict(concurrency=list(durations), duration=list(durations.values()))
    config["multi_run"]["num_runs"] = repeats
    config["benchmark"]["artifacts"]["dir"] = str(directory / "artifacts")
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    aiperf = str(Path.home() / "aiperf-venv/bin/aiperf")
    args = [aiperf, "profile", "--config", str(path)]
    (directory / "command.json").write_text(json.dumps(args))
    set_control(root, mode, stage)
    print(f"START {stage} trials={repeats} levels={list(durations)}", flush=True)
    with (directory / "client.log").open("w") as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=2700)
    set_control(root, mode, stage + "-flush")
    flush_server()
    for rank in range(2):
        audit = json.loads((root / f"audit-{stage}-rank{rank}.json").read_text())
        if audit.get(f"{mode}_decisions", 0) != audit.get("decisions") or not audit.get("decisions"):
            raise RuntimeError(f"Invalid {mode} cost coverage, rank {rank}: {audit}")
        if mode == "measured" and audit.get("sha256") != CURVE_SHA256:
            raise RuntimeError("Measured cost provenance mismatch")
    paths = [path for path in directory.rglob("profile_export_aiperf.json")
             if "phases" not in path.parts and path.with_name("profile_export.jsonl").exists()]
    rows = [load_cell(path, stage, mode) for path in sorted(paths)]
    if len(rows) != len(durations) * repeats:
        raise RuntimeError(f"Expected {len(durations) * repeats} cells in {stage}, got {len(rows)}")
    (directory / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"DONE {stage} cells={len(rows)} min_window={min(row['window_s'] for row in rows):.1f}s", flush=True)
    return rows


def load_cell(path: Path, stage: str, mode: str) -> dict:
    data = json.loads(path.read_text())
    server = json.loads(path.with_name("server_metrics_export.json").read_text())
    records = [json.loads(line) for line in path.with_name("profile_export.jsonl").read_text().splitlines()]
    done = [r for r in records if "output_sequence_length" in r.get("metrics", {})]
    phase = next(p for p in data["input_config"]["phases"] if p["name"] == "profiling")
    c = int(phase["concurrency"])
    metrics = server["metrics"]
    trials = [int(match[1]) for part in path.parts if (match := re.fullmatch(r"(?:trial|run)_(\d+)", part))]
    drafts = build_report.server_total(metrics, "vllm:spec_decode_num_drafts")
    row = dict(mode=mode, stage=stage, concurrency=c, trial=trials[-1] if trials else 1,
               duration=phase.get("duration"), artifact=str(path.parent), included=True,
               requests=len(done), requests_target=int(data["request_count"]["avg"]),
               cancelled=sum(r["metadata"].get("was_cancelled", False) for r in records),
               error_rate=data["request_error_rate"]["avg"],
               preemptions=build_report.server_total(metrics, "vllm:num_preemptions"),
               server_running_p50=build_report.server_stat(metrics, "vllm:num_requests_running", "p50"),
               server_waiting_p50=build_report.server_stat(metrics, "vllm:num_requests_waiting", "p50"),
               acceptance_length=1 + build_report.server_total(metrics, "vllm:spec_decode_num_accepted_tokens") / drafts if drafts else None,
               aiperf_throughput=data["output_token_throughput"]["avg"],
               mean_input_tokens=data["input_sequence_length"]["avg"],
               mean_output_tokens=data["output_sequence_length"]["avg"],
               aiperf_summary_json=json.dumps(data), aiperf_server_metrics_json=json.dumps(server),
               **build_report.steady_window(done, c))
    row.update(build_report.gates(row))
    return row


def reject_errors(rows: list[dict]) -> None:
    failed = [dict(stage=r["stage"], c=r["concurrency"], gates=r["failed_gates"])
              for r in rows if set(filter(None, r["failed_gates"].split(","))) - {"window_ge_5s"}]
    if failed:
        raise RuntimeError(f"Serving gates failed: {failed}")


def validate_rows(rows: list[dict]) -> None:
    reject_errors(rows)
    if any(row["window_s"] < 15 for row in rows):
        raise RuntimeError("Steady window shorter than 15 seconds after duration adjustment")


def set_control(root: Path, mode: str, phase: str) -> None:
    path = root / "control.tmp"
    path.write_text(json.dumps(dict(mode=mode, phase=phase, curve=str(root / "measured-curves-frozen.json"),
                                    sha256=CURVE_SHA256, trace=False)))
    path.replace(root / "control.json")


def flush_server() -> None:
    data = dict(model="qwen", messages=[dict(role="user", content="Hi")], max_tokens=1,
                temperature=0, chat_template_kwargs=dict(enable_thinking=False))
    request = Request("http://127.0.0.1:8676/v1/chat/completions", json.dumps(data).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        response.read()


if __name__ == "__main__":
    main()
