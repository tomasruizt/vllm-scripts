"""Compare extra graph shapes on memory-constrained H100 deployments."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(os.environ.get("ROOT", "/data/tomasruizt/runs/capture-worst-case-20261020"))
PYTHON = os.environ.get("EXPERIMENT_PYTHON", str(Path.home() / ".venv/bin/python"))
CLI = [PYTHON, "-c", "from vllm.entrypoints.cli.main import main; main()"]
MODELS = {"dense": ("Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137"),
          "gdn": ("Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654")}
ADDITIONS = [1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096]


def main() -> None:
    case = sys.argv[1]
    port = 8790 if case == "dense" else 8791
    name, revision = MODELS[case]
    model = Path("/data/tomasruizt/hub_cache") / f"models--Qwen--{name}" / "snapshots" / revision
    directory = ROOT / case
    directory.mkdir(exist_ok=True)
    verify_weights(model)
    subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv"], stdout=(directory / "gpus.csv").open("w"), check=True)
    for startup, variant in [(1, "default"), (1, "4k"), (2, "4k"), (2, "default")]:
        if not run_arm(directory, model, port, startup, variant):
            if variant == "default" and startup == 1:
                break
    (directory / "finished.txt").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def run_arm(directory: Path, model: Path, port: int, startup: int, variant: str) -> bool:
    output = directory / f"s{startup}-{variant}"
    output.mkdir()
    command = [PYTHON, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model), "--served-model-name", "target",
               "--language-model-only", "--dtype", "bfloat16", "--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.9",
               "--max-model-len", "8192", "--max-num-seqs", "128", "--max-num-batched-tokens", "8192",
               "--no-enable-prefix-caching", "--disable-uvicorn-access-log", "--cudagraph-metrics", "--host", "127.0.0.1", "--port", str(port)]
    if variant == "4k":
        info = json.loads((directory / "s1-default/server-info.json").read_text())
        base = info["vllm_config"]["compilation_config"]["cudagraph_capture_sizes"]
        sizes = sorted(set(base) | set(ADDITIONS))
        command += ["--compilation-config", json.dumps({"cudagraph_capture_sizes": sizes})]
    (output / "server-command.json").write_text(json.dumps(command))
    (output / "start-ns.txt").write_text(str(time.time_ns()))
    with (output / "server.log").open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        wait_ready(server, port)
        (output / "ready-ns.txt").write_text(str(time.time_ns()))
        (output / "server-info.json").write_bytes(fetch(port, "server_info?config_format=json", timeout=180))
        print(f"READY {directory.name} s{startup}-{variant}", flush=True)
        for concurrency, input_len in [(1, 1024), (8, 4096), (16, 4096)]:
            measure(output, model, port, concurrency, input_len)
        (output / "arm-exit-code.txt").write_text("0\n")
        return True
    except Exception as error:
        (output / "failure.txt").write_text(repr(error))
        (output / "arm-exit-code.txt").write_text("1\n")
        print(f"FAILED {directory.name} s{startup}-{variant}: {error}", flush=True)
        return False
    finally:
        stop(server)


def measure(output: Path, model: Path, port: int, concurrency: int, input_len: int) -> None:
    cell = output / f"c{concurrency}-i{input_len}"
    cell.mkdir()
    calibration = bench(cell, model, port, concurrency, input_len, max(concurrency, 2), "calibration")
    count = math.ceil(30 / max(calibration["duration"], 0.001) * max(concurrency, 2))
    count = min(128, max(concurrency, math.ceil(count / concurrency) * concurrency))
    bench(cell, model, port, concurrency, input_len, count, "measurement")


def bench(cell: Path, model: Path, port: int, concurrency: int, input_len: int, count: int, name: str) -> dict:
    command = [*CLI, "bench", "serve", "--backend", "vllm", "--base-url", f"http://127.0.0.1:{port}",
               "--endpoint", "/v1/completions", "--model", "target", "--tokenizer", str(model), "--dataset-name", "random",
               "--random-input-len", str(input_len), "--random-output-len", "128", "--random-range-ratio", "0.0",
               "--num-prompts", str(count), "--max-concurrency", str(concurrency), "--request-rate", "inf", "--ignore-eos",
               "--seed", "42", "--save-result", "--save-detailed", "--result-dir", str(cell), "--result-filename", name + ".json",
               "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99"]
    (cell / (name + "-command.json")).write_text(json.dumps(command))
    (cell / (name + "-before.prom")).write_bytes(fetch(port, "metrics"))
    with (cell / (name + ".log")).open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=300)
    (cell / (name + "-after.prom")).write_bytes(fetch(port, "metrics"))
    if result.returncode:
        raise RuntimeError(f"benchmark failed: {cell}/{name}.log")
    data = json.loads((cell / (name + ".json")).read_text())
    if data["completed"] != count:
        raise RuntimeError(f"incomplete requests: {data['completed']}/{count}")
    print(f"BENCH {cell} {name} requests={count} duration={data['duration']:.1f}s", flush=True)
    return data


def verify_weights(model: Path) -> None:
    index = json.loads((model / "model.safetensors.index.json").read_text())
    for name in set(index["weight_map"].values()):
        if not (model / name).is_file():
            raise FileNotFoundError(model / name)


def fetch(port: int, path: str, timeout: int = 30) -> bytes:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/{path}", timeout=timeout) as response:
        return response.read()


def wait_ready(server: subprocess.Popen, port: int) -> None:
    deadline = time.monotonic() + 1200
    while server.poll() is None and time.monotonic() < deadline:
        try:
            fetch(port, "health", timeout=2)
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"server failed or startup timed out, exit={server.poll()}")


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
