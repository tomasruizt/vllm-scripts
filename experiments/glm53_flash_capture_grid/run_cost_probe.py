"""Run a small cost-gap screen using the original real-text AIPerf workload."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT = Path(os.environ["ROOT"])
PORT = 8788
MODEL = os.environ.get("MODEL", "/data/engine/hub_cache/models--zai-org--GLM-5.3-Flash/snapshots/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a")
SMOKE = os.environ.get("SMOKE") == "1"
PYTHON = str(Path.home() / ".venv/bin/python")
AIPERF = str(Path.home() / "aiperf-venv/bin/aiperf")


def main() -> None:
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    os.chdir(ROOT)
    baseline = json.loads((ROOT / "baseline-server-info.json").read_text())
    sizes = baseline["vllm_config"]["compilation_config"]["cudagraph_capture_sizes"]
    assert max(sizes) == 1024, sizes
    sizes = sorted(set(sizes) | {1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096})
    command = [PYTHON, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL,
               "--served-model-name", "glm53-flash", "--tensor-parallel-size", os.environ.get("TP", "4"), "--kv-cache-dtype", "fp8",
               "--speculative-config", '{"method":"mtp","num_speculative_tokens":5}',
               "--enable-auto-tool-choice", "--tool-call-parser", "glm47", "--reasoning-parser", "glm47",
               "--disable-uvicorn-access-log", "--cudagraph-metrics", "--host", "127.0.0.1", "--port", str(PORT),
               "--worker-extension-cls", "cost_probe.CostProbeExtension",
               "--compilation-config", json.dumps({"cudagraph_capture_sizes": sizes})]
    if SMOKE:
        command += ["--max-model-len", "4096", "--max-num-seqs", "32", "--max-num-batched-tokens", "4096",
                    "--gpu-memory-utilization", "0.5", "--language-model-only"]
    (ROOT / "server-command.json").write_text(json.dumps(command))
    (ROOT / "start-ns.txt").write_text(str(time.time_ns()))
    with (ROOT / "server.log").open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    fields = "timestamp,index,uuid,temperature.gpu,clocks.current.sm,utilization.gpu,power.draw,power.limit"
    with (ROOT / "gpu-telemetry.csv").open("w") as log:
        telemetry = subprocess.Popen(["nvidia-smi", "--query-gpu=" + fields, "--format=csv", "-l", "1"],
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    result = 1
    try:
        wait_ready(server)
        (ROOT / "ready-ns.txt").write_text(str(time.time_ns()))
        (ROOT / "server-info.json").write_bytes(fetch("server_info?config_format=json"))
        rpc = {"method": "capture_cost_capabilities", "timeout": 120}
        (ROOT / "probe-capabilities.json").write_bytes(fetch("collective_rpc", rpc))
        print("READY", flush=True)
        for phase in ("native", "probe1", "probe2"):
            measure(phase)
        if SMOKE:
            verify_smoke()
        result = 0
    finally:
        stop(server)
        stop(telemetry)
        (ROOT / "exit-code.txt").write_text(str(result) + "\n")


def measure(phase: str) -> None:
    directory = ROOT / phase
    directory.mkdir()
    text = (ROOT / "benchmark.yaml").read_text()
    replacements = {"__PORT__": PORT, "__MODEL__": MODEL, "__ARTIFACTS__": directory / "artifacts",
                    "__CONCURRENCIES__": 8 if SMOKE else 128, "__DURATIONS__": 10 if SMOKE else 30}
    if SMOKE:
        text = text.replace("max_completion_tokens: 1024", "max_completion_tokens: 64")
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, str(value))
    config = directory / "benchmark.yaml"
    config.write_text(text)
    command = [AIPERF, "profile", "--config", str(config)]
    (directory / "client-command.json").write_text(json.dumps(command))
    with (directory / "validation.log").open("w") as log:
        subprocess.run([AIPERF, "config", "validate", str(config)], stdout=log, stderr=subprocess.STDOUT, check=True)
    reset = json.loads(fetch("reset_prefix_cache", {}))
    assert reset["success"], reset
    if phase != "native":
        rpc = {"method": "capture_cost_probe", "args": [str(ROOT), phase], "timeout": 120}
        (directory / "enable-rpc.json").write_bytes(fetch("collective_rpc", rpc))
    print("CELL " + phase, flush=True)
    with (directory / "client.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=240, check=True)
    if phase != "native":
        rpc = {"method": "capture_cost_probe", "args": [str(ROOT), "off"], "timeout": 120}
        (directory / "disable-rpc.json").write_bytes(fetch("collective_rpc", rpc))
    print("CELL_DONE " + phase, flush=True)


def verify_smoke() -> None:
    paths = sorted(ROOT.glob("probe*-rank*.jsonl"))
    assert len(paths) == 2 * int(os.environ["TP"]), paths
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert {row["mode"] for row in rows} >= {"graph", "bypass"}, path
        assert all((row["graph_replays"] > 0) == (row["mode"] == "graph") for row in rows), path
    for phase in ("native", "probe1", "probe2"):
        summary = next((ROOT / phase).rglob("profile_export_aiperf.json"))
        assert json.loads(summary.read_text())["request_error_rate"]["avg"] == 0, summary
    (ROOT / "smoke-passed.json").write_text(json.dumps(dict(model=MODEL, phases=3, traces=len(paths))))
    print("SMOKE_PASSED", flush=True)


def fetch(endpoint: str, payload: dict | None = None) -> bytes:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}/{endpoint}", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=150) as response:
        return response.read()


def wait_ready(server: subprocess.Popen) -> None:
    deadline = time.monotonic() + 1200
    while server.poll() is None and time.monotonic() < deadline:
        try:
            fetch("health")
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"Server failed or startup timed out: {server.poll()}")


def stop(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def terminate(signum: int, frame: object) -> None:
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    main()
