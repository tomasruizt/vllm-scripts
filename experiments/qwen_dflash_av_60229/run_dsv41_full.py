"""K5 serving calibration and interleaved, audited AIPerf cost-policy comparison."""

import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from urllib.request import Request, urlopen

import yaml

from av_full_curve import freeze_curves
from run_av_full_sweep import load_cell, reject_errors, validate_rows

LEVELS = (1, 8, 32, 64, 128, 256)
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
DATASET = "/data/tomasruizt/benchmarks/deepseek-v4-dep4-av/dataset/throughput_1k.jsonl"
CAPTURES = [1, 2, 4, *range(8, 249, 8), *range(256, 1025, 16), *range(1280, 4097, 256), 5120, 6144]
SMOKE = os.environ.get("DSV41_SMOKE") == "1"
REQUESTS = 8 if SMOKE else 256


def main() -> None:
    root = Path(sys.argv[1])
    port = sys.argv[2]
    model = sys.argv[3]
    if "--prepare" in sys.argv:
        prepare(root, model)
        return
    config = configuration(port)
    source = os.environ.get("DSV41_CALIBRATION_SOURCE")
    if source:
        if (Path(source) / "gpus.txt").read_text() != (root / "gpus.txt").read_text():
            raise RuntimeError("Calibration reuse requires the same physical GPUs in the same order")
        metadata = json.loads((Path(source) / "native-costs-rank0.json").read_text())
        if metadata["configured_drafter_k"] != 5 or metadata["max_num_seqs"] != 256 or metadata["profile_context_len"] != 8192:
            raise RuntimeError("Calibration source configuration mismatch")
        (root / "calibration-source.txt").write_text(source + "\n")
        digest = hashlib.sha256((Path(source) / "timings-rank0.jsonl").read_bytes()).hexdigest()
        (root / "calibration-source.sha256").write_text(digest + "\n")
    else:
        for repeat in ((1,) if SMOKE else (1, 2)):
            for c in ((1, 8) if SMOKE else (1, 8, 32, 128, 256)):
                cell(root, port, config, "calibrate", f"calibrate-r{repeat}-c{c}", c, 3 if SMOKE else 30)
    calibrate_tail(root, port, model)
    flush(root, port, "native", "calibration-flush")
    rows = [json.loads(line) for line in (root / "timings-rank0.jsonl").read_text().splitlines()]
    if source:
        rows += [json.loads(line) for line in (Path(source) / "timings-rank0.jsonl").read_text().splitlines()]
    curve = freeze_curves(rows, CAPTURES, requests=REQUESTS, tokens=16384, steps=5, tensor_parallel_size=1 if SMOKE else 4)
    path = root / "measured-curves-frozen.json"
    path.write_text(json.dumps(curve, indent=2) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "curve.sha256").write_text(digest + "\n")
    print(f"FROZEN {digest} samples={len(rows)}", flush=True)
    compare(root, port, config)
    message = "Smoke workflow passed, dummy weights, no performance conclusions.\n" if SMOKE else "72 interleaved K5 runs passed serving, steady-window and four-rank coverage gates.\n"
    (root / "DONE").write_text(message)
    print("FULL_SWEEP_DONE", flush=True)


def compare(root: Path, port: str, config: dict) -> None:
    levels = (1, 8) if SMOKE else LEVELS
    durations = dict.fromkeys(levels, 3 if SMOKE else 30)
    rows = []
    for repeat in range(1, 3 if SMOKE else 7):
        for c in levels:
            order = ("native", "measured") if repeat % 2 else ("measured", "native")
            pair = [cell(root, port, config, mode, f"r{repeat}-c{c}-{mode}", c, durations[c]) for mode in order]
            if not SMOKE:
                reject_errors(pair)
            window = min(row["window_s"] for row in pair)
            if not SMOKE and repeat == 1 and window < 20:
                durations[c] = math.ceil(30 + 25 - window)
                for row in pair:
                    row["included"] = False
                rows.extend(pair)
                save_metrics(root, rows)
                pair = [cell(root, port, config, mode, f"r{repeat}-c{c}-{mode}-extended", c, durations[c]) for mode in order]
            if not SMOKE:
                validate_rows(pair)
            rows.extend(pair)
            save_metrics(root, rows)
    included = [row for row in rows if row["included"]]
    if not SMOKE:
        validate_rows(included)
    expected = 8 if SMOKE else 72
    if len(included) != expected:
        raise RuntimeError(f"Expected {expected} included runs, got {len(included)}")


def cell(root: Path, port: str, template: dict, mode: str, phase: str, concurrency: int, duration: int) -> dict:
    directory = root / "cells" / phase
    directory.mkdir(parents=True, exist_ok=False)
    config = json.loads(json.dumps(template))
    config["benchmark"]["phases"][0].update(concurrency=concurrency, duration=duration)
    config["benchmark"]["artifacts"]["dir"] = str(directory / "artifacts")
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    args = [str(Path.home() / "aiperf-venv/bin/aiperf"), "profile", "--config", str(path)]
    control(root, mode, phase)
    execute(args, directory, client_environment())
    flush(root, port, mode, phase + "-flush")
    summary = directory / "artifacts/profile_export_aiperf.json"
    data = json.loads(summary.read_text())
    if data["request_error_rate"]["avg"] != 0:
        raise RuntimeError(f"Request errors: {phase}")
    if mode == "calibrate":
        return {}
    audit_phase(root, phase, mode)
    row = load_cell(summary, phase, mode)
    print(f"METRICS {phase} valid={row['valid']} window={row['window_s']:.1f}s throughput={row['throughput']:.1f}", flush=True)
    return row


def configuration(port: str) -> dict:
    config = yaml.safe_load(Path(__file__).with_name("full-curve-sweep.yaml").read_text())
    config.pop("sweep")
    config.pop("multi_run")
    benchmark = config["benchmark"]
    benchmark["model"] = "deepseek"
    benchmark["endpoint"]["url"] = f"http://127.0.0.1:{port}"
    benchmark["endpoint"]["extra"]["chat_template_kwargs"] = dict(thinking=False)
    if SMOKE:
        benchmark["endpoint"]["extra"].update(ignore_eos=True, max_completion_tokens=16, min_tokens=16)
    benchmark["tokenizer"] = dict(name="deepseek-ai/DeepSeek-V4.1-Flash-1017", revision=REVISION)
    benchmark["server_metrics"]["urls"] = [f"http://127.0.0.1:{port}/metrics"]
    return config


def prepare(root: Path, model: str) -> None:
    from transformers import PreTrainedTokenizerFast
    tokenizer_model = os.environ.get("DSV41_TOKENIZER", model)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(Path(tokenizer_model) / "tokenizer.json"))
    with Path(DATASET).open() as handle:
        records = [json.loads(next(handle)) for _ in range(32)]
    text = "\n\n".join(message["content"] for row in records for message in row["messages"] if isinstance(message["content"], str))
    lengths = [1, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 6145, 8192, 10240, 12288, 14336, 16384]
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < max(lengths):
        raise RuntimeError("Real-prompt corpus too short for targeted prefills")
    prompts = []
    for length in lengths:
        prompt = tokenizer.decode(tokens[:length], skip_special_tokens=False)
        if len(tokenizer.encode(prompt, add_special_tokens=False)) != length:
            raise RuntimeError(f"Real-text prefix does not round-trip at {length}")
        prompts.extend(dict(prompt=prompt, output_tokens=1) for _ in range(8))
    directory = root / "tail-calibration"
    directory.mkdir(parents=True)
    (directory / "inputs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in prompts))
    (root / "preflight/prepared-config.yaml").write_text(yaml.safe_dump(configuration("8681"), sort_keys=False))
    print("REAL_TEXT_PREFILLS_READY", flush=True)


def calibrate_tail(root: Path, port: str, model: str) -> None:
    directory = root / "tail-calibration"
    control(root, "calibrate", "calibrate-tail")
    executable = str(Path.home() / "venvs/dsv41-cost-20261021/bin/vllm")
    args = [executable, "bench", "serve", "--backend", "vllm", "--base-url", f"http://127.0.0.1:{port}",
            "--endpoint", "/v1/completions", "--model", model, "--served-model-name", "deepseek",
            "--tokenizer", os.environ.get("DSV41_TOKENIZER", model), "--dataset-name", "custom", "--dataset-path", str(directory / "inputs.jsonl"),
            "--skip-chat-template", "--custom-output-len", "1", "--num-prompts", "112",
            "--max-concurrency", "1", "--ignore-eos", "--extra-body", '{"add_special_tokens":false}',
            "--save-result", "--save-detailed", "--result-dir", str(directory), "--result-filename", "metrics.json"]
    execute(args, directory, client_environment())
    if json.loads((directory / "metrics.json").read_text())["completed"] != 112:
        raise RuntimeError("Incomplete real-text tail calibration")


def execute(args: list[str], directory: Path, env: dict) -> None:
    (directory / "command.txt").write_text(shlex.join(args) + "\n")
    print(f"START {directory.name}", flush=True)
    with (directory / "client.log").open("w") as log:
        subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
    print(f"DONE {directory.name}", flush=True)


def client_environment() -> dict:
    return dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_CACHE="/data/engine/hub_cache")


def control(root: Path, mode: str, phase: str) -> None:
    policy = dict(mode=mode, phase=phase, trace=False)
    if mode == "measured":
        policy.update(curve=str(root / "measured-curves-frozen.json"), sha256=(root / "curve.sha256").read_text().strip())
    temporary = root / "control.tmp"
    temporary.write_text(json.dumps(policy))
    temporary.replace(root / "control.json")


def flush(root: Path, port: str, mode: str, phase: str) -> None:
    control(root, mode, phase)
    data = dict(model="deepseek", messages=[dict(role="user", content="Hello")], max_tokens=1,
                temperature=0, chat_template_kwargs=dict(thinking=False))
    request = Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(data).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        response.read()


def audit_phase(root: Path, phase: str, mode: str) -> None:
    digest = (root / "curve.sha256").read_text().strip()
    for rank in range(1 if SMOKE else 4):
        audit = json.loads((root / f"audit-{phase}-rank{rank}.json").read_text())
        count = audit.get("decisions", 0)
        if not count or audit.get(f"{mode}_decisions") != count or audit["fallback_decisions"]:
            raise RuntimeError(f"Incomplete {mode} coverage: {phase} rank{rank}")
        if mode == "measured" and audit["sha256"] != digest:
            raise RuntimeError(f"Curve checksum mismatch: {phase} rank{rank}")


def save_metrics(root: Path, rows: list[dict]) -> None:
    (root / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


if __name__ == "__main__":
    main()
