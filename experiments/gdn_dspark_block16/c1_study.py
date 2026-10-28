"""Refresh uninstrumented c1 controls, then profile if SD beats the no-SD run."""

import json
import os
import urllib.request
from pathlib import Path
from statistics import median

import yaml

import run as runner


def main() -> None:
    runner.configure_environment()
    exit_code = 1
    try:
        os.environ["BENCH_C1_ONLY"] = "1"
        for mode in os.environ.get("CONTROL_MODES", "fixed adaptive").split():
            runner.validate(mode)
        condition = compare_c1()
        (runner.ROOT / "condition.json").write_text(json.dumps(condition, indent=2))
        if condition["sd_beats_no_sd"]:
            os.environ["PROFILE_C1"] = "1"
            for mode in os.environ.get(
                "PROFILE_MODES", "baseline fixed adaptive"
            ).split():
                profile(mode)
        (runner.ROOT / "DONE").touch()
        exit_code = 0
    finally:
        runner.stop_server()
        (runner.ROOT / "exit-code.txt").write_text(str(exit_code))


def compare_c1() -> dict:
    baseline = Path(os.environ["BASELINE_RUN_ROOT"])
    controls = Path(os.environ.get("CONTROL_RUN_ROOT", str(runner.ROOT)))
    runs = {"baseline": baseline, "fixed": controls, "adaptive": controls}
    tpots = {}
    for mode, root in runs.items():
        paths = sorted(
            (root / mode / "artifacts/profile_runs").rglob("profile_export_aiperf.json")
        )
        data = [json.loads(path.read_text()) for path in paths]
        data = [
            row for row in data if row["input_config"]["phases"][0]["concurrency"] == 1
        ]
        assert len(data) == 3, f"{mode}: expected 3 c1 trials, found {len(data)}"
        assert all(row["request_error_rate"]["avg"] == 0 for row in data)
        tpots[mode] = median(row["inter_token_latency"]["avg"] for row in data)
    return {
        "aiperf_whole_cell_median_tpot_ms": tpots,
        "sd_beats_no_sd": min(tpots["fixed"], tpots["adaptive"]) < tpots["baseline"],
    }


def profile(mode: str) -> None:
    directory = runner.ROOT / "profiles" / mode
    if (directory / "DONE").exists():
        return
    directory.mkdir(parents=True, exist_ok=True)
    runner.status(f"PROFILING {mode}")
    runner.start_server(mode, directory)
    client(directory, "warmup", 2)
    runner.post("/start_profile")
    rpc("begin_c1_profile", directory / "begin.json")
    client(directory, "profile", 8)
    rpc("end_c1_profile", directory / "end.json")
    runner.post("/stop_profile")
    rpc("begin_c1_profile", directory / "events-begin.json", ["events-only"])
    client(directory, "events-only", 8)
    rpc("end_c1_profile", directory / "events-end.json")
    runner.stop_server()
    runner.run(
        [
            "nsys",
            "stats",
            "--report",
            "cuda_gpu_kern_sum,cuda_api_sum,nvtx_gpu_proj_sum",
            "--format",
            "csv",
            "--output",
            str(directory / "stats"),
            str(directory / "trace.nsys-rep"),
        ],
        directory / "nsys-stats.log",
    )
    (directory / "DONE").touch()
    runner.status(f"PROFILE DONE {mode}")


def client(directory: Path, phase: str, requests: int) -> None:
    config = yaml.safe_load((runner.HERE / "sweep.yaml").read_text())
    config.pop("sweep")
    config.pop("multi_run")
    config["benchmark"]["phases"] = [
        {
            "name": "profiling",
            "type": "concurrency",
            "concurrency": 1,
            "requests": requests,
        }
    ]
    config["benchmark"]["tokenizer"]["name"] = runner.TARGET
    config["benchmark"]["endpoint"]["url"] = runner.URL
    config["benchmark"]["server_metrics"]["urls"] = [runner.URL + "/metrics"]
    config["benchmark"]["artifacts"]["dir"] = str(directory / f"{phase}-artifacts")
    path = directory / f"{phase}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    runner.run(
        [
            "env",
            "TMPDIR=/tmp",
            str(Path.home() / "aiperf-venv/bin/aiperf"),
            "profile",
            "--config",
            str(path),
        ],
        directory / f"{phase}.log",
    )
    exports = list(
        (directory / f"{phase}-artifacts").rglob("profile_export_aiperf.json")
    )
    assert len(exports) == 1, f"Expected one {phase} export, found {len(exports)}"
    data = json.loads(exports[0].read_text())
    assert data["request_error_rate"]["avg"] == 0, (
        f"{directory.name} {phase} request errors"
    )


def rpc(method: str, output: Path, args: list[str] | None = None) -> None:
    body = json.dumps({"method": method, "args": args or [], "timeout": 120}).encode()
    request = urllib.request.Request(
        runner.URL + "/collective_rpc",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=150) as response:
        output.write_bytes(response.read())


if __name__ == "__main__":
    main()
