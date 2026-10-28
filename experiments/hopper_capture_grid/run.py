"""Repeat the original AIPerf SPEED-Bench workload on H100 deployments."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(os.environ["ROOT"])
PYTHON = "/data/tomasruizt/capture-env/bin/python"
AIPERF = str(Path.home() / "aiperf-venv/bin/aiperf")
CACHE = Path("/data/tomasruizt/hub_cache")
MODELS = {"dense": ("Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137"),
          "gdn": ("Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654")}
ADDITIONS = [1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096]


def main() -> None:
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    case, variant = sys.argv[1:]
    model_name, revision = MODELS[case]
    model = CACHE / f"models--Qwen--{model_name}" / "snapshots" / revision
    port = int(os.environ.get("PORT_BASE", "8800")) + 2 * list(MODELS).index(case) + (variant == "4k")
    directory = ROOT / case / variant
    directory.mkdir(parents=True)
    result = 1
    try:
        for startup in range(1, 4):
            run_startup(directory, model, port, startup)
        result = 0
    finally:
        (directory / "exit-code.txt").write_text(f"{result}\n")


def run_startup(directory: Path, model: Path, port: int, startup: int) -> None:
    output = directory / f"startup-{startup}"
    output.mkdir()
    command = server_command(directory, model, port)
    (output / "gpu-ids.txt").write_text(os.environ.get("CUDA_VISIBLE_DEVICES", "") + "\n")
    with (output / "gpu-topology.txt").open("w") as topology:
        subprocess.run(["nvidia-smi", "topo", "-m"], stdout=topology, check=True)
    fields = "index,uuid,pstate,clocks.current.sm,clocks.current.memory,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used"
    telemetry_command = ["nvidia-smi", "--query-gpu=timestamp," + fields, "--format=csv", "--loop-ms=1000"]
    with (output / "gpu-telemetry.csv").open("w") as telemetry_log:
        telemetry = subprocess.Popen(telemetry_command, stdout=telemetry_log, stderr=subprocess.STDOUT, start_new_session=True)
    (output / "server-command.json").write_text(json.dumps(command))
    (output / "start-ns.txt").write_text(str(time.time_ns()))
    with (output / "server.log").open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        wait_ready(server, port)
        (output / "ready-ns.txt").write_text(str(time.time_ns()))
        info = fetch(port, "server_info?config_format=json", timeout=180)
        (output / "server-info.json").write_bytes(info)
        print(f"READY {directory.parent.name}/{directory.name}/{startup}", flush=True)
        if os.environ.get("STOP_AT_PREEMPTIONS") == "1":
            peer = "4k" if directory.name == "default" else "default"
            wait_file(directory.parent / peer / output.name / "server-info.json")
        levels = concurrency_levels(directory, startup)
        measured = []
        for concurrency in levels:
            measure(output, model, port, concurrency)
            measured.append(concurrency)
            if os.environ.get("STOP_AT_PREEMPTIONS") == "1":
                pressure = stop_at_pressure(output, concurrency)
                if startup == 1 and pressure:
                    break
        if os.environ.get("STOP_AT_PREEMPTIONS") == "1" and startup == 1:
            (directory / "levels.json").write_text(json.dumps(measured))
        (output / "exit-code.txt").write_text("0\n")
    except Exception as error:
        (output / "failure.txt").write_text(repr(error))
        (output / "exit-code.txt").write_text("1\n")
        raise
    finally:
        stop(server)
        stop(telemetry)


def server_command(directory: Path, model: Path, port: int) -> list[str]:
    command = [PYTHON, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model), "--served-model-name", "qwen38",
               "--language-model-only", "--dtype", "bfloat16", "--tensor-parallel-size", os.environ.get("TP", "1"), "--gpu-memory-utilization", "0.90",
               "--max-model-len", "8192", "--max-num-seqs", "256", "--max-num-batched-tokens", "16384",
               "--enable-prefix-caching", "--kv-cache-dtype", "fp8", "--reasoning-parser", "qwen3",
               "--disable-uvicorn-access-log", "--cudagraph-metrics", "--host", "127.0.0.1", "--port", str(port)]
    if directory.parent.name == "gdn":
        command += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": 3})]
    if directory.name == "4k":
        baseline = directory.parent / "default/startup-1/server-info.json"
        wait_file(baseline)
        config = json.loads(baseline.read_text())["vllm_config"]["compilation_config"]
        sizes = sorted(set(config["cudagraph_capture_sizes"]) | set(ADDITIONS))
        command += ["--compilation-config", json.dumps({"cudagraph_capture_sizes": sizes})]
    return command


def measure(output: Path, model: Path, port: int, concurrency: int) -> None:
    cell = output / f"c{concurrency}"
    cell.mkdir()
    text = (ROOT / "benchmark.yaml").read_text()
    replacements = {"__PORT__": port, "__MODEL__": model, "__ARTIFACTS__": cell / "artifacts",
                    "__CONCURRENCIES__": concurrency, "__DURATIONS__": 60 if concurrency == 256 else 30}
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, str(value))
    config = cell / "benchmark.yaml"
    config.write_text(text)
    subprocess.run([AIPERF, "config", "validate", str(config)], stdout=(cell / "validation.log").open("w"), stderr=subprocess.STDOUT, check=True)
    reset = fetch(port, "reset_prefix_cache", method="POST")
    (cell / "cache-reset.json").write_bytes(reset)
    assert json.loads(reset)["success"], reset
    command = [AIPERF, "profile", "--config", str(config)]
    (cell / "client-command.json").write_text(json.dumps(command))
    print(f"CELL {cell.relative_to(ROOT)}", flush=True)
    with (cell / "client.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=240, check=True)
    assert list((cell / "artifacts").rglob("profile_export_aiperf.json")), cell
    summary = json.loads(next((cell / "artifacts").rglob("profile_export_aiperf.json")).read_text())
    metrics = json.loads(next((cell / "artifacts").rglob("server_metrics_export.json")).read_text())["metrics"]
    preemptions = sum(series["stats"]["total"] for series in metrics["vllm:num_preemptions"]["series"])
    status = dict(preemptions=preemptions, error_rate=summary["request_error_rate"]["avg"])
    temporary = cell / "status.tmp"
    temporary.write_text(json.dumps(status))
    temporary.replace(cell / "status.json")
    print(f"CELL_DONE {cell.relative_to(ROOT)}", flush=True)


def concurrency_levels(directory: Path, startup: int) -> list[int]:
    selected = os.environ.get("CONCURRENCIES")
    if selected:
        levels = [int(value) for value in selected.split(",")]
        (directory / "levels.json").write_text(json.dumps(levels))
        return levels
    if os.environ.get("STOP_AT_PREEMPTIONS") == "1":
        return json.loads((directory / "levels.json").read_text()) if startup > 1 else [1, 8, 32, 64, 128, 256]
    return [256, 128, 64, 32, 8, 1] if startup != 2 else [1, 8, 32, 64, 128, 256]


def stop_at_pressure(output: Path, concurrency: int) -> bool:
    variant = "4k" if output.parent.name == "default" else "default"
    peer = output.parents[1] / variant / output.name / f"c{concurrency}" / "status.json"
    wait_file(peer)
    own = output / f"c{concurrency}" / "status.json"
    statuses = [json.loads(path.read_text()) for path in (own, peer)]
    return any(status["preemptions"] > 0 or status["error_rate"] > 0 for status in statuses)


def wait_file(path: Path) -> None:
    deadline = time.monotonic() + 1200
    while not path.exists():
        worker = next(parent for parent in path.parents if parent.name in ("default", "4k"))
        if (worker / "exit-code.txt").exists():
            raise RuntimeError(f"Peer worker exited before producing artifact: {path}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"Missing peer artifact: {path}")
        time.sleep(2)


def terminate(signum: int, frame: object) -> None:
    raise SystemExit(128 + signum)


def fetch(port: int, endpoint: str, timeout: int = 30, method: str = "GET") -> bytes:
    request = urllib.request.Request(f"http://127.0.0.1:{port}/{endpoint}", method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def wait_ready(server: subprocess.Popen, port: int) -> None:
    deadline = time.monotonic() + 1200
    while server.poll() is None and time.monotonic() < deadline:
        try:
            fetch(port, "health", timeout=2)
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"Server failed or startup timed out: {server.poll()}")


def stop(server: subprocess.Popen) -> None:
    try:
        os.killpg(server.pid, signal.SIGTERM)
        server.wait(timeout=30)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()


if __name__ == "__main__":
    main()
