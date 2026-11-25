"""One chain on one GPU: serve each variant in turn and run the workloads against it.

Variants differ only in the vLLM tree (`PYTHONPATH` overlay of the main install) and flags.
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(os.environ["ROOT"])
TREES = Path.home() / "code/av-sync"
VLLM_PYTHON = str(TREES / "main/.venv/bin/python")
AIPERF = str(Path.home() / ".venv/bin/aiperf")
MODELS = {
    "4b": ("Qwen/Qwen3-4B", {"model": "RedHatAI/Qwen3-4B-speculator.dspark", "method": "dspark", "num_speculative_tokens": 5}),
    "27b": ("Qwen/Qwen3.5-27B", {"model": "z-lab/Qwen3.5-27B-DFlash", "method": "dflash", "num_speculative_tokens": 7}),
}


def cached(repo: str) -> str:
    """The snapshot path of a cached repo that has no `refs/main` (offline lookups by name fail)."""
    cache = Path.home() / ".cache/huggingface/hub" / f"models--{repo.replace('/', '--')}"
    return repo if (cache / "refs/main").exists() else str(next((cache / "snapshots").iterdir()))


MODEL, SPEC = MODELS[os.environ.get("MODEL_KEY", "4b")]
MODEL_PATH, SPEC = cached(MODEL), SPEC | {"model": cached(SPEC["model"])}
PORT = int(os.environ["PORT"])
GPUS = os.environ["CUDA_VISIBLE_DEVICES"]
MAX_NUM_SEQS = os.environ.get("MAX_NUM_SEQS", "16")
AV = SPEC | {"enable_adaptive_verification": True}
# variant -> (tree, speculative config, extra serve args)
VARIANTS = {
    "k5": ("main", SPEC, []),
    "av_main": ("main", AV, []),
    "av_depth3": ("main", AV, ["--async-scheduling-depth", "3"]),
    "av_ring3": ("ring3", AV, []),  # main with a 3-slot stale ring at depth 2
    "av_lazy": ("lazy", AV, []),  # vllm#63355 merged with main
}
# Synthetic workloads: (input tokens, output tokens, concurrency, trials), 30 s each.
SYNTHETIC = {"prefill": (8192, 16, 1, 2), "decode": (512, 256, 16, 1), "decode_c1": (512, 512, 1, 2)}
# Chat workloads replay the mixed RedHatAI/speculator_benchmarks prompts in order (same requests for
# every variant): `chat_c<N>` is closed loop at concurrency N, `chat_r<rate>` Poisson arrivals.
PROMPTS = str(ROOT / "prompts/mixed.jsonl")
SELECTED = os.environ.get("WORKLOADS", "prefill decode").split()


def main() -> None:
    signal.signal(signal.SIGTERM, terminate)
    for variant in sys.argv[1:]:
        run_variant(variant, ROOT / f"gpu{GPUS.replace(',', '+')}" / variant)


def run_variant(variant: str, output: Path) -> None:
    output.mkdir(parents=True)
    tree, spec, extra_args = VARIANTS[variant]
    env = os.environ | {"PYTHONPATH": str(TREES / tree)}
    command = server_command(spec, extra_args)
    (output / "server-command.json").write_text(json.dumps(command))
    which = subprocess.run([VLLM_PYTHON, "-c", "import vllm; print(vllm.__file__)"], env=env, capture_output=True, text=True)
    (output / "vllm-path.txt").write_text(which.stdout)
    with (output / "server.log").open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    try:
        wait_ready(server)
        for workload in SELECTED:
            load_args, trials = workload_args(workload)
            for trial in range(1, trials + 1):
                aiperf(output / f"{workload}-trial-{trial}", load_args, seed=trial)
    finally:
        stop(server)


def server_command(spec: dict, extra_args: list[str]) -> list[str]:
    return [VLLM_PYTHON, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL_PATH, "--served-model-name", MODEL,
            "--max-model-len", "16384", "--gpu-memory-utilization", "0.85", "--max-num-seqs", MAX_NUM_SEQS,
            "--tensor-parallel-size", str(len(GPUS.split(","))),
            "--no-enable-prefix-caching", "--speculative-config", json.dumps(spec),
            "--disable-uvicorn-access-log", "--host", "127.0.0.1", "--port", str(PORT), *extra_args]


def workload_args(name: str) -> tuple[list[str], int]:
    """AIPerf load arguments and the number of trials of a workload."""
    if name in SYNTHETIC:
        input_tokens, output_tokens, concurrency, trials = SYNTHETIC[name]
        return ["--tokenizer", MODEL_PATH, "--synthetic-input-tokens-mean", str(input_tokens), "--synthetic-input-tokens-stddev", "0",
                "--output-tokens-mean", str(output_tokens), "--output-tokens-stddev", "0",
                "--extra-inputs", "ignore_eos:true", "--concurrency", str(concurrency),
                "--benchmark-duration", "30", "--warmup-request-count", str(3 * concurrency)], trials
    chat = ["--input-file", PROMPTS, "--custom-dataset-type", "single_turn", "--dataset-sampling-strategy", "sequential",
            "--extra-inputs", "max_tokens:1024"]
    if name.startswith("chat_c"):
        concurrency = int(name.removeprefix("chat_c"))
        requests = max(32, 8 * concurrency)
        return [*chat, "--concurrency", str(concurrency), "--request-count", str(requests),
                "--warmup-request-count", str(concurrency)], 1
    rate = float(name.removeprefix("chat_r"))
    return [*chat, "--request-rate", str(rate), "--arrival-pattern", "poisson",
            "--request-count", str(round(60 * rate)), "--warmup-request-count", str(round(5 * rate))], 2


def aiperf(cell: Path, load_args: list[str], seed: int) -> None:
    cell.mkdir(parents=True)
    command = [AIPERF, "profile", "--model", MODEL, "--url", f"http://127.0.0.1:{PORT}",
               "--endpoint-type", "chat", "--streaming", *load_args, "--extra-inputs", "temperature:0",
               "--use-server-token-count", "--random-seed", str(seed),
               "--server-metrics", f"http://127.0.0.1:{PORT}/metrics", "--ui-type", "none",
               "--artifact-dir", str(cell / "artifacts")]
    (cell / "client-command.json").write_text(json.dumps(command))
    with (cell / "client.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=True)


def terminate(signum: int, frame: object) -> None:
    raise SystemExit(128 + signum)


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
