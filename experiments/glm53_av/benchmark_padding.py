"""Benchmark DEP4 DSpark with an empty prefix cache per run."""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

PYTHON = "/home/tomasruizt/.venv/bin/python"
COUNTERS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-patch", type=Path, required=True)
    parser.add_argument("--av", choices=["on", "off"], default="on")
    parser.add_argument("--require-budget-expansion", action="store_true")
    parser.add_argument(
        "--cache", type=Path, default=Path("/data/tomasruizt/huggingface/hub")
    )
    parser.add_argument("--port", type=int, default=8026)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument(
        "--concurrencies", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256]
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=args.repo)
    if patch != args.expected_patch.read_bytes():
        raise RuntimeError("Worktree changed while queued; update the queued job first")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    command = server_command(args)
    env = os.environ | {
        "PATH": str(Path(PYTHON).parent) + ":" + os.environ["PATH"],
        "PYTHONPATH": str(args.repo),
        "HF_HUB_CACHE": str(args.cache),
        "HF_HUB_OFFLINE": "1",
        "CUDA_HOME": "/usr/local/cuda",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
    }
    result = {
        "command": command,
        "repo": str(args.repo),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.repo, text=True
        ).strip(),
        "source_patch": str(args.expected_patch),
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "prefix_caching": True,
        "adaptive_verification": args.av == "on",
        "reset_prefix_cache_before_each_run": True,
        "num_runs": args.num_runs,
        "concurrencies": args.concurrencies,
        "runs": [],
        "status": "starting",
    }
    save(args, result)
    started = time.monotonic()
    with (args.output / "server.log").open("w") as log:
        server = subprocess.Popen(
            command,
            cwd=args.repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        (args.output / "server.pid").write_text(str(server.pid))
        try:
            wait_for_server(args, server)
            result["server_startup_seconds"] = time.monotonic() - started
            reset_prefix_cache(args)
            print("Server healthy; warming up", flush=True)
            evaluate(args, "warmup", 32, 64, 512, env)
            time.sleep(10)
            for repeat in range(1, args.num_runs + 1):
                for concurrency in args.concurrencies:
                    label = f"gsm8k-c{concurrency}-r{repeat}"
                    reset = reset_prefix_cache(args)
                    (args.output / f"{label}-cache-reset.json").write_text(
                        json.dumps(reset) + "\n"
                    )
                    before = metrics(args, f"{label}-metrics-before.txt")
                    print(f"Running {label} after successful cache reset", flush=True)
                    run = evaluate(args, label, concurrency, 1319, 2048, env)
                    time.sleep(10)
                    after = metrics(args, f"{label}-metrics-after.txt")
                    delta = {key: after[key] - before[key] for key in COUNTERS}
                    drafts, drafted, accepted = (delta[key] for key in COUNTERS)
                    if drafts <= 0 or drafted <= 0:
                        raise RuntimeError(
                            f"Missing speculative decoding activity: {delta}"
                        )
                    run.update(
                        concurrency=concurrency,
                        repeat=repeat,
                        mean_acceptance_length=1 + accepted / drafts,
                        draft_acceptance_rate=accepted / drafted,
                        counter_deltas=delta,
                        prefix_cache_reset=True,
                    )
                    (args.output / f"{label}.json").write_text(
                        json.dumps(run, indent=2) + "\n"
                    )
                    result["runs"].append(run)
                    result["status"] = "running"
                    save(args, result)
                    print(json.dumps(run), flush=True)
            result["budget_expansion_seen"] = (
                "AV reclaimed DP graph padding:"
                in (args.output / "server.log").read_text()
            )
            if args.require_budget_expansion and not result["budget_expansion_seen"]:
                raise RuntimeError("No successful AV budget expansion was logged")
            result["status"] = "completed"
        except BaseException as exc:
            result.update(status="failed", error=str(exc))
            raise
        finally:
            save(args, result)
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


def server_command(args):
    target = "RedHatAI/GLM-5.3-NVFP4"
    draft = "RedHatAI/GLM-5.3-speculator.dspark"
    target_revision = "c8917e4258572c405575855ff53effe58c17a38e"
    draft_revision = "b374b95663447ea0e935151be4f3d6666e36e6d7"
    for model, revision in ((target, target_revision), (draft, draft_revision)):
        snapshot = (
            args.cache
            / ("models--" + model.replace("/", "--"))
            / "snapshots"
            / revision
        )
        if not snapshot.is_dir():
            raise FileNotFoundError(snapshot)
    spec = {
        "method": "dspark",
        "model": draft,
        "revision": draft_revision,
        "num_speculative_tokens": 8,
        "attention_backend": "FLASH_ATTN",
        "draft_sample_method": "probabilistic",
        "enable_adaptive_verification": args.av == "on",
    }
    return [
        PYTHON,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        target,
        "--revision",
        target_revision,
        "--served-model-name",
        "glm-5.3",
        "--tensor-parallel-size",
        "1",
        "--data-parallel-size",
        "4",
        "--enable-expert-parallel",
        "--all2all-backend",
        "allgather_reducescatter",
        "--attention-backend",
        "FLASHINFER_MLA_SPARSE",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--block-size",
        "64",
        "--max-model-len",
        "16384",
        "--max-num-seqs",
        "128",
        "--max-num-batched-tokens",
        "16384",
        "--gpu-memory-utilization",
        "0.85",
        "--reasoning-parser",
        "glm45",
        "--chat-template-content-format",
        "string",
        "--trust-remote-code",
        "--disable-uvicorn-access-log",
        "--enable-prefix-caching",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--compilation-config",
        json.dumps(
            {
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "max_cudagraph_capture_size": 1152,
            }
        ),
        "--speculative-config",
        json.dumps(spec),
    ]


def wait_for_server(args, server):
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"Server exited with code {server.returncode}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{args.port}/health", timeout=2
            ):
                return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(5)
    raise TimeoutError("Server startup exceeded one hour")


def reset_prefix_cache(args):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        request = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/reset_prefix_cache", method="POST"
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        if result.get("success") is True:
            return result
        time.sleep(2)
    raise RuntimeError("Prefix cache reset failed; refusing to benchmark a warm cache")


def evaluate(args, label, concurrency, questions, max_tokens, env):
    output = args.output / f"{label}.json"
    log_path = args.output / f"{label}-client.log"
    with log_path.open("w") as log:
        subprocess.run(
            [
                PYTHON,
                "tests/evals/gsm8k/gsm8k_eval.py",
                "--port",
                str(args.port),
                "--num-questions",
                str(questions),
                "--num-shots",
                "5",
                "--max-tokens",
                str(max_tokens),
                "--temperature",
                "0",
                "--seed",
                "42",
                "--max-concurrency",
                str(concurrency),
                "--request-timeout-seconds",
                "1800",
                "--save-results",
                str(output),
            ],
            cwd=args.repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=1800,
        )
    if "Error calling vLLM" in log_path.read_text():
        raise RuntimeError(f"Failed API requests in {log_path}")
    return json.loads(output.read_text())


def metrics(args, filename):
    with urllib.request.urlopen(
        f"http://127.0.0.1:{args.port}/metrics", timeout=30
    ) as response:
        text = response.read().decode()
    (args.output / filename).write_text(text)
    values = dict.fromkeys(COUNTERS, 0.0)
    for line in text.splitlines():
        match = re.match(
            r"(vllm:spec_decode_num_(?:drafts|draft_tokens|accepted_tokens)_total)(?:\{[^}]*\})?\s+(\S+)",
            line,
        )
        if match:
            key, value = match.groups()
            values[key] += float(value)
    return values


def save(args, result):
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
