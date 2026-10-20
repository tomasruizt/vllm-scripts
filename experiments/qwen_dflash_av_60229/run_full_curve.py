"""Collect real costs, freeze a complete curve, run measured-only c256 cells."""

import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

from av_full_curve import freeze_curves
from cost_oracle_queue import run_cell


def main() -> None:
    root = Path(sys.argv[1])
    port = sys.argv[2]
    template = Path(__file__).with_name("steady-speedbench.yaml").read_text()
    calibration = [dict(phase=f"calibrate-r{repeat}-c{c}", mode="calibrate", concurrency=c, requests=n)
                   for repeat in range(1, 3) for c, n in ((1, 24), (8, 96), (32, 192), (128, 512), (256, 1024))]
    for command in calibration:
        execute(root, port, template, command)
    calibrate_tail(root, port)
    # Flush asynchronous GPU events without synchronizing timed steps.
    execute(root, port, template, dict(phase="calibrate-tail-flush", mode="calibrate", concurrency=1, requests=8))
    rows = [json.loads(line) for line in (root / "timings-rank0.jsonl").read_text().splitlines()]
    captures = [1, 2, 4, *range(8, 249, 8), *range(256, 1025, 16), *range(1280, 4097, 256), 5120, 6144]
    curve = root / "measured-curves-frozen.json"
    curve.write_text(json.dumps(freeze_curves(rows, captures), indent=2) + "\n")
    digest = hashlib.sha256(curve.read_bytes()).hexdigest()
    (root / "curve.sha256").write_text(digest + "\n")
    print(f"FROZEN {digest}, samples={len(rows)}", flush=True)
    for repeat in range(1, 7):
        command = dict(phase=f"measured-r{repeat}-c256", mode="measured", concurrency=256,
                       requests=1024, curve=str(curve), sha256=digest)
        execute(root, port, template, command)
    command = dict(phase="measured-trace-c256", mode="measured", concurrency=256,
                   requests=1024, curve=str(curve), sha256=digest, trace=True)
    execute(root, port, template, command)
    execute(root, port, template, dict(command, phase="measured-flush", concurrency=1, requests=8, trace=False))
    for phase in [f"measured-r{repeat}-c256" for repeat in range(1, 7)] + ["measured-trace-c256"]:
        for rank in range(2):
            audit = json.loads((root / f"audit-{phase}-rank{rank}.json").read_text())
            if (audit["sha256"] != digest or audit.get("measured_decisions", 0) == 0
                    or audit["measured_decisions"] != audit["decisions"] or audit["fallback_decisions"]):
                raise RuntimeError(f"Invalid measured-only coverage: {phase}, rank{rank}")
    (root / "DONE").write_text("All measured-only runs completed, inspect AIPerf validity before analysis.\n")
    print("FULL_CURVE_DONE", flush=True)


def execute(root: Path, port: str, template: str, command: dict) -> None:
    path = root / "commands" / f"{command['phase']}.json"
    if path.with_suffix(".done").exists():
        return
    path.write_text(json.dumps(command))
    run_cell(root, port, template, path)
    summary = root / "cells" / command["phase"] / "artifacts/profile_export_aiperf.json"
    data = json.loads(summary.read_text())
    if data["request_error_rate"]["avg"] != 0:
        raise RuntimeError(f"AIPerf request errors: {command['phase']}")


def calibrate_tail(root: Path, port: str) -> None:
    from transformers import AutoTokenizer
    import numpy as np

    model = "/data/tomasruizt/hub_cache/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754"
    directory = root / "tail-calibration"
    directory.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    rng = np.random.default_rng(542)
    lengths = [1024, 1280, 1536, 1792, 2048, 2304, 2560, 3072, 3584, 4096,
               5120, 6144, 6145, 8192, 10240, 12288, 14336, 16377]
    rows = [dict(prompt=exact_prompt(tokenizer, rng, length), output_tokens=1)
            for length in lengths for _ in range(8)]
    dataset = directory / "inputs.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    temporary = root / "control.tmp"
    temporary.write_text(json.dumps(dict(mode="calibrate", phase="calibrate-tail")))
    temporary.replace(root / "control.json")
    args = [str(Path.home() / ".venv/bin/vllm"), "bench", "serve", "--backend", "vllm",
            "--base-url", f"http://127.0.0.1:{port}", "--endpoint", "/v1/completions",
            "--model", model, "--served-model-name", "qwen", "--tokenizer", model,
            "--dataset-name", "custom", "--dataset-path", str(dataset), "--skip-chat-template",
            "--custom-output-len", "1",
            "--num-prompts", str(len(rows)), "--max-concurrency", "1", "--ignore-eos",
            "--save-result", "--save-detailed", "--result-dir", str(directory),
            "--result-filename", "metrics.json"]
    (directory / "command.txt").write_text(shlex.join(args) + "\n")
    print("START calibrate-tail", flush=True)
    with (directory / "client.log").open("w") as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
    result = json.loads((directory / "metrics.json").read_text())
    if result["completed"] != len(rows):
        raise RuntimeError("Incomplete targeted tail measurements")
    print("DONE calibrate-tail", flush=True)


def exact_prompt(tokenizer: object, rng: object, length: int) -> str:
    # Decode randomized vocabulary IDs, then repair tokenization boundaries.
    text = tokenizer.decode(rng.integers(1000, tokenizer.vocab_size - 1000, size=2 * length))
    for _ in range(10):
        text = tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)[:length])
        count = len(tokenizer.encode(text, add_special_tokens=False))
        if count == length:
            return text
        text += " a" * (length - count)
    raise RuntimeError(f"Cannot construct exact {length}-token prompt")


if __name__ == "__main__":
    main()
