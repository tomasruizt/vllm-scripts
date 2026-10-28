"""Measure TTFT and capture a torch profile of one spec-decode variant on the small DSV4-Flash-DSpark model."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(os.environ["ROOT"])
VLLM_PYTHON = str(Path.home() / "code/vllm-dsv41/.venv/bin/python")
AIPERF = str(Path.home() / "aiperf-venv/bin/aiperf")
REPO = "inference-optimization/DeepSeek-V4-Flash-DSpark-2.1B-A1.9B-small"
MODEL = os.environ["HF_HUB_CACHE"] + f"/models--{REPO.replace('/', '--')}/snapshots/bbddd703cacc556881da865ecb67663454e943ab"
SPEC = {"nospec": None,
        "k5": {"method": "dspark", "num_speculative_tokens": 5, "draft_sample_method": "probabilistic"},
        "k5av": {"method": "dspark", "num_speculative_tokens": 5, "draft_sample_method": "probabilistic",
                 "enable_adaptive_verification": True}}
SPEC["k5av_nosync"] = SPEC["k5av"]  # with VLLM_DIAG_SKIP_AV_CONF_SYNC=1 (diagnostic patch on the node)
SPEC["k5av_lazysync"] = SPEC["k5av"]  # vLLM from the worktree with the lazy-sync fix
VARIANT_ENV = {"k5av_nosync": {"VLLM_DIAG_SKIP_AV_CONF_SYNC": "1"},
               "k5av_lazysync": {"PYTHONPATH": str(Path.home() / "code/vllm-av-lazy-sync")}}
PORT = int(os.environ["PORT"])


def main() -> None:
    signal.signal(signal.SIGTERM, terminate)
    variant = sys.argv[1]
    output = ROOT / variant
    output.mkdir(parents=True)
    command = server_command(variant, output / "torch-traces")
    (output / "server-command.json").write_text(json.dumps(command))
    (output / "gpu-ids.txt").write_text(os.environ.get("CUDA_VISIBLE_DEVICES", "") + "\n")
    env = os.environ | VARIANT_ENV.get(variant, {})
    with (output / "server.log").open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    try:
        wait_ready(server)
        for trial in (1, 2, 3):
            aiperf(output / f"ttft-trial-{trial}", ["--benchmark-duration", "30", "--warmup-request-count", "3"], seed=trial)
        post("start_profile")
        aiperf(output / "profiled", ["--request-count", "5"], seed=99)
        post("stop_profile")
        (output / "exit-code.txt").write_text("0\n")
    except BaseException as error:
        (output / "failure.txt").write_text(repr(error))
        (output / "exit-code.txt").write_text("1\n")
        raise
    finally:
        stop(server)


def server_command(variant: str, trace_dir: Path) -> list[str]:
    profiler = {"profiler": "torch", "torch_profiler_dir": str(trace_dir), "torch_profiler_with_stack": False}
    command = [VLLM_PYTHON, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL, "--served-model-name", REPO,
               "--load-format", "dummy", "--max-model-len", "16384", "--gpu-memory-utilization", "0.85",
               "--max-num-seqs", "16", "--no-enable-prefix-caching",
               "--compilation-config", json.dumps({"max_cudagraph_capture_size": 4096}),
               "--profiler-config", json.dumps(profiler),
               "--disable-uvicorn-access-log", "--host", "127.0.0.1", "--port", str(PORT)]
    if SPEC[variant]:
        command += ["--speculative-config", json.dumps(SPEC[variant])]
    return command


def aiperf(cell: Path, stop_args: list[str], seed: int) -> None:
    cell.mkdir(parents=True)
    command = [AIPERF, "profile", "--model", REPO, "--tokenizer", REPO, "--url", f"http://127.0.0.1:{PORT}",
               "--endpoint-type", "chat", "--streaming", "--prompt-corpus", "random", "--random-corpus-style", "vllm",
               "--synthetic-input-tokens-mean", "8192", "--synthetic-input-tokens-stddev", "0",
               "--output-tokens-mean", "16", "--output-tokens-stddev", "0", "--extra-inputs", "ignore_eos:true",
               "--use-server-token-count", "--concurrency", "1", "--random-seed", str(seed),
               "--server-metrics", f"http://127.0.0.1:{PORT}/metrics", "--ui-type", "none",
               "--artifact-dir", str(cell / "artifacts"), *stop_args]
    (cell / "client-command.json").write_text(json.dumps(command))
    with (cell / "client.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)


def terminate(signum: int, frame: object) -> None:
    raise SystemExit(128 + signum)


def post(endpoint: str) -> None:
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}/{endpoint}", method="POST")
    urllib.request.urlopen(request, timeout=600).read()


def wait_ready(server: subprocess.Popen) -> None:
    deadline = time.monotonic() + 1800
    while server.poll() is None and time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read()
            return
        except Exception:
            time.sleep(3)
    raise RuntimeError(f"Server failed or startup timed out: {server.poll()}")


def stop(server: subprocess.Popen) -> None:
    try:
        os.killpg(server.pid, signal.SIGTERM)
        server.wait(timeout=120)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()


if __name__ == "__main__":
    main()
