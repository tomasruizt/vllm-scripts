"""Start one DSV4.1-Flash server variant and run its validation phases for PR #59986."""
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
SERVED = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
MODEL = os.environ["HF_HUB_CACHE"] + "/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"
TP = os.environ.get("TP", "2")
PORT = int(os.environ.get("PORT", "8820"))
K3 = {"method": "dspark", "num_speculative_tokens": 3}
K5AV = {"method": "dspark", "num_speculative_tokens": 5, "draft_sample_method": "probabilistic",
        "rejection_sample_method": "block", "enable_adaptive_verification": True}
FULL = {"cudagraph_mode": "FULL"}
FULL_AND_PIECEWISE = {"mode": "VLLM_COMPILE", "cudagraph_mode": "FULL_AND_PIECEWISE"}
ATTENTION = {"backend": "FLASHINFER_MLA_SPARSE_DSV41", "indexer_kv_dtype": "fp8"}
FULL_ARGS = ["--attention-config", json.dumps(ATTENTION), "--no-enable-prefix-caching"]
# The #59970 serve flags, without autotune since this arm only checks stability.
AV_ARGS = ["--attention-config", json.dumps(ATTENTION | {"indexer_kv_dtype": "mxfp4", "indexer_sparse_logits": True}),
           "--enable-prefix-caching", "--long-prefill-token-threshold", "4096", "--no-enable-flashinfer-autotune"]
# name: (source tree, compilation config or None for eager, speculative config, extra server args, phases)
VARIANTS = {
    "base-full": ("base", FULL, K3, FULL_ARGS, ["probes", "gsm8k", "perf"]),
    "pr-full": ("pr", FULL, K3, FULL_ARGS, ["probes", "gsm8k", "perf"]),
    "eager": ("base", None, K3, FULL_ARGS, ["probes", "gsm8k"]),
    "base-av": ("base", FULL_AND_PIECEWISE, K5AV, AV_ARGS, ["load", "probes"]),
    "pr-av": ("pr", FULL_AND_PIECEWISE, K5AV, AV_ARGS, ["load", "probes"]),
}
PERF_CONCURRENCIES = [1, 8, 32, 64]
PERF_DURATION_S = 45
LOAD_DURATION_S = int(os.environ.get("LOAD_DURATION_S", "600"))


def main() -> None:
    signal.signal(signal.SIGTERM, terminate)
    variant = sys.argv[1]
    tree, _, _, _, phases = VARIANTS[variant]
    src = Path.home() / f"code/vllm-59986-{tree}"
    output = ROOT / variant
    output.mkdir(parents=True)
    command = server_command(variant)
    (output / "server-command.json").write_text(json.dumps(command))
    (output / "gpu-ids.txt").write_text(os.environ.get("CUDA_VISIBLE_DEVICES", "") + "\n")
    (output / "start-ns.txt").write_text(str(time.time_ns()))
    env = os.environ | {"PYTHONPATH": str(src)}
    with (output / "server.log").open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        wait_ready(server)
        (output / "ready-ns.txt").write_text(str(time.time_ns()))
        (output / "server-info.json").write_bytes(fetch("server_info?config_format=json", timeout=180))
        for phase in phases:
            print(f"PHASE {variant}/{phase}", flush=True)
            PHASES[phase](output / phase, src)
        (output / "server-alive.txt").write_text(f"{server.poll() is None}\n")
        (output / "exit-code.txt").write_text("0\n")
    except BaseException as error:
        (output / "failure.txt").write_text(repr(error))
        (output / "exit-code.txt").write_text("1\n")
        raise
    finally:
        stop(server)


def server_command(variant: str) -> list[str]:
    _, compilation, spec, extra_args, _ = VARIANTS[variant]
    command = [VLLM_PYTHON, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL,
               "--served-model-name", SERVED, "--tensor-parallel-size", TP, "--enable-expert-parallel",
               "--tokenizer-mode", "deepseek_v41", "--reasoning-parser", "deepseek_v41",
               "--tool-call-parser", "deepseek_v41", "--enable-auto-tool-choice",
               "--max-model-len", "262144", "--gpu-memory-utilization", "0.88", "--max-num-seqs", "128",
               "--max-num-batched-tokens", "8192", "--kv-cache-dtype", "fp8", "--block-size", "128",
               "--engram-config", json.dumps({"cpu_offload": True}),
               "--speculative-config", json.dumps(spec), *extra_args,
               "--disable-uvicorn-access-log", "--host", "127.0.0.1", "--port", str(PORT)]
    command += ["--compilation-config", json.dumps(compilation)] if compilation else ["--enforce-eager"]
    return command


def run_probes(output: Path, src: Path) -> None:
    """The #62417 corruption probes: greedy chat, thinking on, one request at a time."""
    output.mkdir(parents=True)
    results = [{"probe": name} | chat(body) for name, body in PROBES.items()]
    (output / "probes.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))


def run_gsm8k(output: Path, src: Path) -> None:
    output.mkdir(parents=True)
    command = [VLLM_PYTHON, str(src / "tests/evals/gsm8k/gsm8k_eval.py"), "--use-chat-completions",
               "--model", SERVED, "--port", str(PORT), "--num-questions", "200", "--max-tokens", "8192",
               "--temperature", "0", "--max-concurrency", "32", "--save-results", str(output / "gsm8k.json")]
    run_client(output, command, timeout=3600)


def run_perf(output: Path, src: Path) -> None:
    for concurrency in PERF_CONCURRENCIES:
        cell = output / f"c{concurrency}"
        cell.mkdir(parents=True)
        workload = ["--synthetic-input-tokens-mean", "1024", "--synthetic-input-tokens-stddev", "0",
                    "--output-tokens-mean", "256", "--output-tokens-stddev", "0",
                    "--extra-inputs", "ignore_eos:true", "--extra-inputs", "reasoning_effort:none",
                    "--concurrency", str(concurrency), "--warmup-request-count", str(2 * concurrency),
                    "--benchmark-duration", str(PERF_DURATION_S), "--random-seed", str(concurrency)]
        run_client(cell, aiperf_command(cell, workload), timeout=900)


def run_load(output: Path, src: Path) -> None:
    """The #59970 load: closed-loop multi-turn chat at c96, so prefix-cached prefills keep joining decode batches."""
    output.mkdir(parents=True)
    workload = ["--conversation-num", "2000", "--conversation-turn-mean", "6", "--conversation-turn-stddev", "3",
                "--shared-system-prompt-length", "600",
                "--synthetic-input-tokens-mean", "2048", "--synthetic-input-tokens-stddev", "2048",
                "--output-tokens-mean", "600", "--output-tokens-stddev", "800",
                "--extra-inputs", "ignore_eos:true", "--concurrency", "96",
                "--benchmark-duration", str(LOAD_DURATION_S), "--random-seed", "59970"]
    run_client(output, aiperf_command(output, workload), timeout=LOAD_DURATION_S + 900, check=False)


PHASES = {"probes": run_probes, "gsm8k": run_gsm8k, "perf": run_perf, "load": run_load}


def aiperf_command(cell: Path, workload: list[str]) -> list[str]:
    # AIPerf rejects a local snapshot path offline, but resolves the repo id from HF_HUB_CACHE.
    return [AIPERF, "profile", "--model", SERVED, "--tokenizer", SERVED, "--url", f"http://127.0.0.1:{PORT}",
            "--endpoint-type", "chat", "--streaming", "--prompt-corpus", "random", "--random-corpus-style", "vllm",
            "--use-server-token-count", "--benchmark-grace-period", "120",
            "--server-metrics", f"http://127.0.0.1:{PORT}/metrics", "--ui-type", "none",
            "--artifact-dir", str(cell / "artifacts"), *workload]


def run_client(cell: Path, command: list[str], timeout: int, check: bool = True) -> None:
    (cell / "client-command.json").write_text(json.dumps(command))
    with (cell / "client.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=check)
    (cell / "client-exit-code.txt").write_text(f"{result.returncode}\n")


TOOLS = [{"type": "function", "function": {
    "name": "get_gpu_alert", "description": "Get the current GPU alert status of a cluster node.",
    "parameters": {"type": "object", "properties": {"node": {"type": "string"}}, "required": ["node"]}}}]
PROBES = {
    "math_short": {"messages": [{"role": "user", "content": "17*19=? answer with the number only"}]},
    "capital": {"messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}]},
    "cjk": {"messages": [{"role": "user", "content": "请用三句话简要介绍长城的历史。"}]},
    "tools": {"messages": [{"role": "user", "content": "Is there a GPU alert on node gpu-03?"}], "tools": TOOLS},
    "chat_scaffold": {"messages": [
        {"role": "system", "content": "You are a concise assistant for a GPU cluster operations team."},
        {"role": "user", "content": "Our nightly training job failed with NCCL timeouts on two nodes."},
        {"role": "assistant", "content": "Which nodes, and did anything change in the network config?"},
        {"role": "user", "content": "gpu-03 and gpu-07, nothing changed. List three things to check, one line each."}]},
}


def chat(body: dict) -> dict:
    request = body | {"model": SERVED, "temperature": 0, "max_tokens": 2048}
    data = json.loads(fetch("v1/chat/completions", timeout=600, data=request))
    choice = data["choices"][0]
    message = choice["message"]
    return {"finish_reason": choice["finish_reason"], "content": message.get("content"),
            "reasoning": message.get("reasoning") or message.get("reasoning_content"),
            "tool_calls": message.get("tool_calls"), "completion_tokens": data["usage"]["completion_tokens"]}


def terminate(signum: int, frame: object) -> None:
    raise SystemExit(128 + signum)


def fetch(endpoint: str, timeout: int = 30, data: dict | None = None) -> bytes:
    payload = json.dumps(data).encode() if data is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}/{endpoint}", data=payload, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def wait_ready(server: subprocess.Popen) -> None:
    deadline = time.monotonic() + 3 * 3600
    while server.poll() is None and time.monotonic() < deadline:
        try:
            fetch("health", timeout=2)
            return
        except Exception:
            time.sleep(5)
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
