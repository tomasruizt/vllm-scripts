"""Compare DEP4 AV with and without graph-padding reclamation using AIPerf."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "glm53_av"))
from benchmark_padding import (
    COUNTERS,
    PYTHON,
    metrics,
    reset_prefix_cache,
    wait_for_server,
)

MODEL = "deepseek-ai/DeepSeek-V4-Flash-DSpark"
REVISION = "62af8fffb2f7030cac4de2f0169f5b8d1101b646"
AIPERF = "/tmp/dep4-aiperf-venv/bin/aiperf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-repo", type=Path, required=True)
    parser.add_argument("--reclaim-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8034)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--methods", nargs="+", choices=["avon", "reclaim"], default=["avon", "reclaim"]
    )
    parser.add_argument(
        "--concurrencies", nargs="+", type=int, default=[8, 16, 32, 64, 128, 256]
    )
    args = parser.parse_args()
    for repo, expected in [
        (args.baseline_repo, "58b3298457dde7b4554b3b4e20b238c0ac2c3a65"),
        (args.reclaim_repo, "b39cfba61"),
    ]:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        assert commit.startswith(expected), (repo, commit)
        assert not subprocess.check_output(["git", "diff", "HEAD"], cwd=repo)
    records = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    assert len(records) == 1536
    assert all(len(row["messages"]) == 1 for row in records)
    assert all(
        "FULL BENCHMARK DATA" not in row["messages"][0]["content"] for row in records
    )
    for mode, repo in [("avon", args.baseline_repo), ("reclaim", args.reclaim_repo)]:
        if mode in args.methods:
            run_mode(args, mode, repo, records)


def run_mode(args, mode, repo, records, *, repeats=None, workload=None, prepare=None):
    original_dataset = args.dataset
    output = args.output / mode
    output.mkdir(parents=True, exist_ok=True)
    previous = None
    if (output / "result.json").exists() and args.resume:
        previous = json.loads((output / "result.json").read_text())
    elif (output / "result.json").exists():
        raise RuntimeError(f"Results already exist: {output}")
    args.output = output
    env = os.environ | {
        "PATH": str(Path(PYTHON).parent) + ":" + os.environ["PATH"],
        "PYTHONPATH": str(repo),
        "HF_HOME": "/data/tomasruizt/huggingface",
        "HF_HUB_CACHE": "/data/tomasruizt/huggingface/hub",
        "HF_DATASETS_CACHE": "/data/tomasruizt/huggingface/datasets",
        "HF_XET_CACHE": "/data/tomasruizt/huggingface/xet",
        "HF_HUB_OFFLINE": "1",
        "CUDA_HOME": "/usr/local/cuda",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
    }
    env.pop("HF_TOKEN", None)
    env.pop("HUGGING_FACE_HUB_TOKEN", None)
    command = server_command(args.port)
    result = {
        "mode": mode,
        "command": command,
        "repo": str(repo),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "dataset": str(args.dataset),
        "requests_per_run": len(records),
        "output_tokens_per_request": 512,
        "prefix_caching": True,
        "reset_prefix_cache_before_each_run": True,
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "num_runs": args.num_runs,
        "concurrencies": args.concurrencies,
        "status": "starting",
        "runs": [],
    }
    if workload is not None:
        result["requests_per_run"] = None
        result["target_measurement_seconds"] = args.target_seconds
        result["workload_plan"] = str(output.parent / "workload-plan.json")
    if previous is not None:
        for key in ["commit", "dataset_sha256", "num_runs", "concurrencies"]:
            assert result[key] == previous[key], key
        result["runs"] = previous["runs"]
        result["previous_server_startup_seconds"] = previous.get(
            "server_startup_seconds"
        )
    completed = {(run["repeat"], run["concurrency"]) for run in result["runs"]}

    def save():
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    started = time.monotonic()
    with (output / "server.log").open("a" if previous is not None else "w") as log:
        server = subprocess.Popen(
            command,
            cwd=repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_for_server(args, server)
            result["server_startup_seconds"] = time.monotonic() - started
            reset_prefix_cache(args)
            profile(
                args,
                f"warmup-resume-{time.time_ns()}" if previous else "warmup",
                32,
                64,
                env,
            )
            time.sleep(10)
            if prepare is not None:
                prepare(args, env)
            for repeat in repeats or range(1, args.num_runs + 1):
                for concurrency in args.concurrencies:
                    if (repeat, concurrency) in completed:
                        continue
                    label = f"speedbench-c{concurrency}-r{repeat}"
                    request_count = len(records)
                    if workload is not None:
                        args.dataset, request_count = workload(repeat, concurrency)
                    reset = reset_prefix_cache(args)
                    (output / f"{label}-cache-reset.json").write_text(
                        json.dumps(reset) + "\n"
                    )
                    before = metrics(args, f"{label}-metrics-before.txt")
                    print(f"{mode}: {label}, cache reset succeeded", flush=True)
                    summary = profile(args, label, concurrency, request_count, env)
                    time.sleep(5)
                    after = metrics(args, f"{label}-metrics-after.txt")
                    delta = {key: after[key] - before[key] for key in COUNTERS}
                    drafts, drafted, accepted = (delta[key] for key in COUNTERS)
                    assert drafts > 0 and drafted > 0
                    run = {
                        "concurrency": concurrency,
                        "repeat": repeat,
                        "mean_acceptance_length": 1 + accepted / drafts,
                        "draft_acceptance_rate": accepted / drafted,
                        "counter_deltas": delta,
                        "prefix_cache_reset": True,
                        "aiperf_summary": str(summary.relative_to(output)),
                        "request_count": request_count,
                        "dataset": str(args.dataset),
                        "dataset_sha256": hashlib.sha256(
                            args.dataset.read_bytes()
                        ).hexdigest(),
                    }
                    result["runs"].append(run)
                    result["status"] = "running"
                    save()
            seen = (
                "AV reclaimed DP graph padding:" in (output / "server.log").read_text()
            )
            assert seen == (mode == "reclaim"), (mode, seen)
            result.update(
                status="completed"
                if len(result["runs"]) == args.num_runs * len(args.concurrencies)
                else "block_completed",
                budget_expansion_seen=seen,
            )
        except BaseException as exc:
            result.update(status="failed", error=repr(exc))
            raise
        finally:
            save()
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
            args.output = output.parent
            args.dataset = original_dataset
            args.port += 1


def profile(args, label, concurrency, count, env):
    artifact = args.output / label
    if artifact.exists() and args.resume:
        artifact.rename(artifact.with_name(f"{label}-interrupted-{time.time_ns()}"))
    command = [
        AIPERF,
        "profile",
        "--model",
        "deepseek-v4-flash",
        "--url",
        f"http://127.0.0.1:{args.port}",
        "--endpoint-type",
        "chat",
        "--streaming",
        "--use-server-token-count",
        "--tokenizer",
        MODEL,
        "--tokenizer-revision",
        REVISION,
        "--input-file",
        str(args.dataset),
        "--custom-dataset-type",
        "speed_bench_throughput_1k",
        "--dataset-sampling-strategy",
        "sequential",
        "--random-seed",
        "42",
        "--concurrency",
        str(concurrency),
        "--request-count",
        str(count),
        "--osl",
        "512",
        "--extra-inputs",
        '{"temperature":0,"ignore_eos":true,"min_tokens":512}',
        "--output-artifact-dir",
        str(artifact),
    ]
    (args.output / f"{label}-command.json").write_text(
        json.dumps(command, indent=2) + "\n"
    )
    client_env = env.copy()
    client_env.pop("PYTHONPATH", None)
    with (args.output / f"{label}-client.log").open("w") as log:
        subprocess.run(
            command,
            cwd=args.output,
            env=client_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=max(180, args.target_seconds * 4)
            if hasattr(args, "target_seconds")
            else 7200,
        )
    summary = artifact / "profile_export_aiperf.json"
    report = json.loads(summary.read_text())
    assert report.get("is_complete") is not False
    assert report["request_count"]["avg"] == count
    assert not (report.get("error_request_count") or {}).get("avg", 0)
    assert (
        report["output_sequence_length"]["min"]
        == report["output_sequence_length"]["max"]
        == 512
    )
    return summary


def server_command(port):
    spec = {
        "method": "dspark",
        "model": MODEL,
        "revision": REVISION,
        "num_speculative_tokens": 7,
        "draft_sample_method": "greedy",
        "enable_adaptive_verification": True,
    }
    return [
        PYTHON,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        MODEL,
        "--revision",
        REVISION,
        "--served-model-name",
        "deepseek-v4-flash",
        "--tokenizer-mode",
        "deepseek_v4",
        "--trust-remote-code",
        "--tensor-parallel-size",
        "1",
        "--data-parallel-size",
        "4",
        "--enable-expert-parallel",
        "--all2all-backend",
        "allgather_reducescatter",
        "--moe-backend",
        "deep_gemm_mega_moe",
        "--kv-cache-dtype",
        "fp8",
        "--block-size",
        "256",
        "--max-model-len",
        "16384",
        "--max-num-seqs",
        "128",
        "--max-num-batched-tokens",
        "16384",
        "--gpu-memory-utilization",
        "0.85",
        "--attention-config",
        '{"indexer_kv_dtype":"mxfp4"}',
        "--enable-prefix-caching",
        "--disable-uvicorn-access-log",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--compilation-config",
        '{"cudagraph_mode":"FULL_AND_PIECEWISE","max_cudagraph_capture_size":1152}',
        "--speculative-config",
        json.dumps(spec),
    ]


if __name__ == "__main__":
    main()
