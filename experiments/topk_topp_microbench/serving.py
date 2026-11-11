"""Controlled Qwen-only serving A/B, one server and interleaved methods."""

import argparse
import json
import signal
import urllib.request
from pathlib import Path

from capture import make_level


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    level = make_level(output)
    extra = level.config["request_extra"]
    assert extra["top_k"] == 20 and extra["top_p"] == 0.8 and extra["temperature"] == 0.7
    signal.signal(signal.SIGTERM, lambda *_: exit(1))
    try:
        level.start_server([], level.root / "server", level.config["serve_args"])
        smoke = coding_smoke(level, output)
        (output / "coding-smoke.json").write_text(json.dumps(smoke, indent=2))
        cells = []
        for repeat in range(3):
            modes = ["native", "flashinfer-deterministic"]
            if repeat % 2:
                modes.reverse()
            for mode in modes:
                level.rpc("set_filter_mode", [mode])
                for concurrency in (1, 32):
                    cells.append(measure_cell(level, output, repeat, mode, concurrency))
                    (output / "cells.json").write_text(json.dumps(cells, indent=2))
    finally:
        level.stop_server()
    (output / "DONE").write_text("ok\n")


def measure_cell(level, output: Path, repeat: int, mode: str, concurrency: int) -> dict:
    directory = output / f"r{repeat}-{mode}-c{concurrency}"
    directory.mkdir()
    client = level.aiperf(directory, "measurement", duration=30, concurrency=concurrency)
    code = client.wait(timeout=180)
    row = dict(repeat=repeat, mode=mode, concurrency=concurrency, directory=directory.name, exit_code=code)
    print(json.dumps(row), flush=True)
    if code:
        raise RuntimeError(f"AIPerf failed: {directory}")
    return row


def coding_smoke(level, output: Path) -> list[dict]:
    prompts = [
        "Write a Python function is_prime(n) with trial division. Return only code.",
        "Write Python binary_search(xs, value), returning the index or -1. Return only code.",
        "Write a JavaScript debounce(fn, delay) function. Return only code.",
        "Write a PostgreSQL query for a running total of sales(amount, sale_date). Return only SQL.",
        "Write Rust gcd(a: u64, b: u64) using Euclid's algorithm. Return only code.",
        "Write pytest parametrized tests for a Python factorial(n) function. Return only code.",
    ]
    rows = []
    for index, prompt in enumerate(prompts):
        responses = {}
        for mode in ("native", "flashinfer-deterministic"):
            level.rpc("set_filter_mode", [mode])
            responses[mode] = request(level, prompt)
        left = responses["native"]["choices"][0]["message"]["content"]
        right = responses["flashinfer-deterministic"]["choices"][0]["message"]["content"]
        row = dict(index=index, prompt=prompt, exact_text=left == right, responses=responses)
        if left != right:
            level.rpc("set_filter_mode", ["native"])
            control = request(level, prompt)
            row["native_control"] = control
            row["native_repeat_exact"] = left == control["choices"][0]["message"]["content"]
        rows.append(row)
        (output / "coding-smoke.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(dict(smoke=index, exact_text=row["exact_text"])), flush=True)
    return rows


def request(level, prompt: str) -> dict:
    body = dict(model=level.served_name(), messages=[dict(role="user", content=prompt)],
                max_completion_tokens=256, **level.config["request_extra"])
    data = json.dumps(body).encode()
    call = urllib.request.Request(level.url + "/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(call, timeout=180) as response:
        return json.load(response)


if __name__ == "__main__":
    main()
