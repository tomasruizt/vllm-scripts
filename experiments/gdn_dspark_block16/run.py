"""Reserved GPU validation. Timed servers have no execution probes."""

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil
import yaml

HERE = Path(__file__).resolve().parent
ROOT = Path(
    os.environ.get(
        "RUN_ROOT", str(Path.home() / "runs/gdn-dspark-block16-20261026-1745")
    )
)
SOURCE = Path(
    os.environ.get(
        "SOURCE_DIR", str(Path.home() / "code/vllm-dspark-block16-20261026-1745")
    )
)
PYTHON = str(Path.home() / ".venv/bin/python")
TARGET = os.environ.get(
    "TARGET_MODEL_DIR",
    "/data/tomasruizt/hub_cache/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
)
DRAFT = os.environ.get(
    "DRAFT_MODEL_DIR",
    "/data/tomasruizt/hub_cache/models--satgeze--Qwen3.6-27B-DSpark/snapshots/47d8de04dd814f4a727bf3fc6a85613d2a3dcaa1",
)
PORT = int(os.environ.get("BENCH_PORT", "8696"))
URL = f"http://127.0.0.1:{PORT}"
SERVER: subprocess.Popen | None = None


def main() -> None:
    configure_environment()
    exit_code = 1
    try:
        run(["ninja", "--version"], ROOT / "build-tools.log")
        if os.environ.get("RUN_METADATA_TESTS", "1") == "1":
            run(
                [
                    PYTHON,
                    "-m",
                    "pytest",
                    str(SOURCE / "tests/v1/attention/test_gdn_metadata_builder.py"),
                    "-q",
                ],
                ROOT / "metadata-tests.log",
            )
        modes = os.environ.get("VALIDATION_MODES", "diagnostic baseline fixed adaptive")
        for mode in modes.split():
            validate(mode)
        (ROOT / "DONE").touch()
        exit_code = 0
    finally:
        stop_server()
        (ROOT / "exit-code.txt").write_text(str(exit_code))


def configure_environment() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(os.environ.get("BENCH_TMPDIR", str(ROOT / "tmp")))
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    os.environ.update(
        PYTHONPATH=f"{SOURCE}:{HERE}",
        HF_HUB_CACHE="/data/tomasruizt/hub_cache",
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_SERVER_DEV_MODE="1",
        VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN="1024",
        TMPDIR=str(temp_dir),
        PATH=f"{Path.home() / '.venv/bin'}:{Path.home() / '.local/bin'}:{os.environ['PATH']}",
    )
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)


def validate(mode: str) -> None:
    directory = ROOT / mode
    directory.mkdir(exist_ok=True)
    status(f"STARTING {mode}")
    start_server(mode, directory)
    status(f"READY {mode}")
    if mode == "diagnostic":
        (directory / "probe-enabled").touch()
        post("/start_profile")
        run(
            [
                PYTHON,
                str(HERE / "workload.py"),
                "--port",
                str(PORT),
                "--output",
                str(directory / "requests.json"),
                "--count",
                "24",
                "--concurrency",
                "16",
                "--stagger",
            ],
            directory / "client.log",
        )
        post("/stop_profile")
        (directory / "probe-enabled").unlink()
    elif not (directory / "gsm8k.json").exists():
        run(
            [
                PYTHON,
                str(HERE / "workload.py"),
                "--port",
                str(PORT),
                "--output",
                str(directory / "gsm8k.json"),
                "--stagger",
            ],
            directory / "gsm8k.log",
        )
    if mode in ("baseline", "fixed", "adaptive"):
        sweep(directory)
    stop_server()
    if mode == "diagnostic":
        run(
            [
                "nsys",
                "stats",
                "--report",
                "cuda_gpu_kern_sum",
                "--format",
                "csv",
                "--output",
                str(directory / "kernels"),
                str(directory / "trace.nsys-rep"),
            ],
            directory / "nsys-stats.log",
        )
    (directory / "DONE").touch()
    status(f"DONE {mode}")


def start_server(mode: str, directory: Path) -> None:
    global SERVER
    try:
        urllib.request.urlopen(URL + "/health", timeout=2)
    except (urllib.error.URLError, TimeoutError):
        pass
    else:
        raise RuntimeError(f"Port {PORT} is occupied")
    sizes = (
        list(range(1, 33))
        + list(range(48, 513, 16))
        + list(range(768, 2049, 256))
        + [3072, 4096]
    )
    command = [
        PYTHON,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        TARGET,
        "--served-model-name",
        "qwen",
        "--language-model-only",
        "--max-model-len",
        "8192",
        "--tensor-parallel-size",
        os.environ.get("TENSOR_PARALLEL_SIZE", "1"),
        "--max-num-seqs",
        "128",
        "--max-num-batched-tokens",
        "8192",
        "--gpu-memory-utilization",
        "0.92",
        "--no-enable-prefix-caching",
        "--disable-uvicorn-access-log",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--compilation-config",
        json.dumps({"cudagraph_capture_sizes": sizes}),
    ]
    if mode != "baseline":
        spec = {
            "method": "dspark",
            "model": DRAFT,
            "num_speculative_tokens": 15,
            "draft_sample_method": "probabilistic",
            "enable_adaptive_verification": mode != "fixed",
        }
        command += ["--speculative-config", json.dumps(spec)]
    env = os.environ.copy()
    profile_c1 = os.environ.get("PROFILE_C1") == "1"
    if mode == "diagnostic" or profile_c1:
        extension = (
            "profile_probe.C1ProfileProbe"
            if profile_c1
            else "execution_probe.ExecutionProbe"
        )
        if profile_c1:
            env["C1_PROFILE_OUTPUT"] = str(directory)
        else:
            env.update(
                EXECUTION_EVIDENCE=str(directory / "execution.jsonl"),
                EXECUTION_PROBE_ENABLE=str(directory / "probe-enabled"),
            )
        command += [
            "--worker-extension-cls",
            extension,
            "--profiler-config",
            '{"profiler":"cuda"}',
        ]
        command = [
            "nsys",
            "profile",
            "--trace",
            "cuda,nvtx",
            "--sample",
            "none",
            "--cpuctxsw",
            "none",
            "--cuda-graph-trace",
            "node",
            "--capture-range",
            "cudaProfilerApi",
            "--capture-range-end",
            "stop",
            "--output",
            str(directory / "trace"),
        ] + command
    (directory / "command.json").write_text(json.dumps(command, indent=2))
    with (directory / "server.log").open("w") as stream:
        SERVER = subprocess.Popen(
            command, env=env, stdout=stream, stderr=subprocess.STDOUT
        )
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        if SERVER.poll() is not None:
            raise RuntimeError(
                f"{mode} server exited {SERVER.returncode}, see {directory / 'server.log'}"
            )
        try:
            urllib.request.urlopen(URL + "/health", timeout=2)
            return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise TimeoutError(f"{mode} startup exceeded 20 minutes")


def sweep(directory: Path) -> None:
    config = yaml.safe_load((HERE / "sweep.yaml").read_text())
    durations = [30, 30, 30, 40] if directory.name == "adaptive" else [30, 30, 45, 60]
    config["sweep"]["parameters"]["duration"] = durations
    if os.environ.get("BENCH_C1_ONLY") == "1":
        config["sweep"]["parameters"] = {"concurrency": [1], "duration": [30]}
    config["benchmark"]["tokenizer"]["name"] = TARGET
    config["benchmark"]["endpoint"]["url"] = URL
    config["benchmark"]["server_metrics"]["urls"] = [URL + "/metrics"]
    config["benchmark"]["artifacts"]["dir"] = str(directory / "artifacts")
    path = directory / "sweep.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    status(f"MEASURING {directory.name}")
    run(
        [
            "env",
            "TMPDIR=/tmp",
            str(Path.home() / "aiperf-venv/bin/aiperf"),
            "profile",
            "--config",
            str(path),
        ],
        directory / "aiperf.log",
        timeout=2400,
    )
    exports = list(
        (directory / "artifacts/profile_runs").rglob("profile_export_aiperf.json")
    )
    expected = 3 if os.environ.get("BENCH_C1_ONLY") == "1" else 12
    assert len(exports) == expected, (
        f"Expected {expected} benchmark cells, found {len(exports)}"
    )


def stop_server() -> None:
    global SERVER
    if SERVER is None:
        return
    try:
        children = psutil.Process(SERVER.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    SERVER.terminate() if SERVER.poll() is None else None
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(children, timeout=20)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        SERVER.wait(timeout=20)
    except subprocess.TimeoutExpired:
        SERVER.kill()
        SERVER.wait()
    SERVER = None


def run(command: list[str], log: Path, timeout: int = 1200) -> None:
    with log.open("w") as stream:
        subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
        )


def post(path: str) -> None:
    with urllib.request.urlopen(
        urllib.request.Request(URL + path, method="POST"), timeout=120
    ) as response:
        assert response.status == 200


def status(message: str) -> None:
    (ROOT / "status.txt").write_text(
        f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}\n"
    )
    print(message, flush=True)


def interrupted(*args) -> None:
    raise SystemExit(143)


if __name__ == "__main__":
    main()
