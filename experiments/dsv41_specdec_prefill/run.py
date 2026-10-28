"""Start one DSV4.1-Flash server variant and measure Robert's random-prompt workload with AIPerf."""
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
MODEL = os.environ["HF_HUB_CACHE"] + "/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"
TP = os.environ.get("TP", "8")
SPEC = {"nospec": None,
        "k3": {"method": "dspark", "num_speculative_tokens": 3, "draft_sample_method": "probabilistic"},
        "k5": {"method": "dspark", "num_speculative_tokens": 5, "draft_sample_method": "probabilistic"},
        "k5av": {"method": "dspark", "num_speculative_tokens": 5, "draft_sample_method": "probabilistic",
                 "enable_adaptive_verification": True}}
SPEC["k5av_lazysync"] = SPEC["k5av"]  # vLLM from the av-lazy-confidence-sync worktree
VARIANT_ENV = {"k5av_lazysync": {"PYTHONPATH": str(Path.home() / "code/vllm-av-lazy-sync")}}
CONCURRENCIES = [1, 16]
TRIALS = 3
DURATION_S = 60
PORT = int(os.environ.get("PORT", "8810"))


def main() -> None:
    signal.signal(signal.SIGTERM, terminate)
    variant = sys.argv[1]
    output = ROOT / variant
    output.mkdir(parents=True)
    command = server_command(variant)
    (output / "server-command.json").write_text(json.dumps(command))
    (output / "gpu-ids.txt").write_text(os.environ.get("CUDA_VISIBLE_DEVICES", "") + "\n")
    (output / "start-ns.txt").write_text(str(time.time_ns()))
    env = os.environ | VARIANT_ENV.get(variant, {})
    with (output / "server.log").open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    try:
        wait_ready(server)
        (output / "ready-ns.txt").write_text(str(time.time_ns()))
        (output / "server-info.json").write_bytes(fetch("server_info?config_format=json", timeout=180))
        for concurrency in CONCURRENCIES:
            for trial in range(1, TRIALS + 1):
                measure(output / f"c{concurrency}" / f"trial-{trial}", concurrency, seed=100 * concurrency + trial)
        (output / "exit-code.txt").write_text("0\n")
    except BaseException as error:
        (output / "failure.txt").write_text(repr(error))
        (output / "exit-code.txt").write_text("1\n")
        raise
    finally:
        stop(server)


def server_command(variant: str) -> list[str]:
    command = [VLLM_PYTHON, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL,
               "--served-model-name", "deepseek-ai/DeepSeek-V4.1-Flash-1017",
               "--tensor-parallel-size", TP, "--enable-expert-parallel",
               "--tokenizer-mode", "deepseek_v41", "--reasoning-parser", "deepseek_v41",
               "--tool-call-parser", "deepseek_v41", "--enable-auto-tool-choice",
               "--max-model-len", "1048576", "--gpu-memory-utilization", "0.88", "--max-num-seqs", "64",
               "--block-size", os.environ.get("BLOCK_SIZE", "64"),  # Blackwell's FLASHMLA_MEGA_ATTN_DSV41 needs 128 "--safetensors-load-strategy", "prefetch", "--enable-ep-weight-filter",
               "--compilation-config", json.dumps({"max_cudagraph_capture_size": 4096}),
               "--engram-config", json.dumps({"cpu_offload": True}),
               "--no-enable-prefix-caching", "--enable-prompt-tokens-details",
               "--disable-uvicorn-access-log", "--host", "127.0.0.1", "--port", str(PORT)]
    if SPEC[variant]:
        command += ["--speculative-config", json.dumps(SPEC[variant])]
    return command


def measure(cell: Path, concurrency: int, seed: int) -> None:
    cell.mkdir(parents=True)
    # AIPerf rejects a local snapshot path offline, but resolves the repo id from HF_HUB_CACHE.
    command = [AIPERF, "profile", "--model", "deepseek-ai/DeepSeek-V4.1-Flash-1017",
               "--tokenizer", "deepseek-ai/DeepSeek-V4.1-Flash-1017",
               "--url", f"http://127.0.0.1:{PORT}", "--endpoint-type", "chat", "--streaming",
               "--prompt-corpus", "random", "--random-corpus-style", "vllm",
               "--synthetic-input-tokens-mean", "8192", "--synthetic-input-tokens-stddev", "0",
               "--output-tokens-mean", "16", "--output-tokens-stddev", "0",
               "--extra-inputs", "ignore_eos:true", "--extra-inputs", "reasoning_effort:none",
               "--use-server-token-count", "--concurrency", str(concurrency),
               "--warmup-request-count", str(2 * concurrency), "--benchmark-duration", str(DURATION_S),
               "--benchmark-grace-period", "120", "--random-seed", str(seed),
               "--server-metrics", f"http://127.0.0.1:{PORT}/metrics", "--ui-type", "none",
               "--artifact-dir", str(cell / "artifacts")]
    (cell / "client-command.json").write_text(json.dumps(command))
    print(f"CELL {cell.relative_to(ROOT)}", flush=True)
    load_before = Path("/proc/loadavg").read_text()
    with (cell / "client.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)
    (cell / "loadavg.txt").write_text(f"before {load_before}after {Path('/proc/loadavg').read_text()}")
    assert list((cell / "artifacts").rglob("profile_export_aiperf.json")), cell


def terminate(signum: int, frame: object) -> None:
    raise SystemExit(128 + signum)


def fetch(endpoint: str, timeout: int = 30) -> bytes:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/{endpoint}", timeout=timeout) as response:
        return response.read()


def wait_ready(server: subprocess.Popen) -> None:
    deadline = time.monotonic() + 3 * 3600  # a cold load from NFS alone takes ~1 h
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
