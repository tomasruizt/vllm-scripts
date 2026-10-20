"""Execute queued standard AIPerf cells while keeping the vLLM server alive."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen


def main() -> None:
    root = Path(sys.argv[1])
    port = sys.argv[2]
    template = Path(sys.argv[3]).read_text()
    plan_path = Path(os.environ.get("AV_COST_PLAN", Path(__file__).with_name("cost-oracle-plan.json")))
    plan = json.loads(plan_path.read_text())
    if os.environ.get("AV_COST_REVERSE_PAIRS") == "1":
        plan = [item for i in range(0, len(plan), 2) for item in reversed(plan[i:i + 2])]
    for index, command in enumerate(plan):
        path = root / "commands" / f"{index:03}.json"
        if not path.exists():
            path.write_text(json.dumps(command))
    while not (root / "STOP").exists():
        commands = sorted((root / "commands").glob("*.json"))
        for path in commands:
            if not path.with_suffix(".done").exists():
                run_cell(root, port, template, path)
        if os.environ.get("AV_COST_STOP_AFTER_PLAN") == "1":
            break
        time.sleep(1)
    flush_trace(root, port)


def flush_trace(root: Path, port: str) -> None:
    control = root / "control.tmp"
    control.write_text(json.dumps(dict(mode="native", phase="flush-c1")))
    control.replace(root / "control.json")
    data = dict(model="qwen", messages=[dict(role="user", content="Hi")],
                max_tokens=1, temperature=0, chat_template_kwargs=dict(enable_thinking=False))
    request = Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(data).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        response.read()


def run_cell(root: Path, port: str, template: str, command_path: Path) -> None:
    command = json.loads(command_path.read_text())
    directory = root / "cells" / command["phase"]
    directory.mkdir(parents=True, exist_ok=False)
    control = root / "control.tmp"
    control.write_text(json.dumps(command))
    control.replace(root / "control.json")
    config = template.replace("__PORT__", port).replace("__C__", str(command["concurrency"]))
    config = config.replace("__N__", str(command["requests"]))
    length = command.get("fixed_output_tokens")
    if length is not None:
        config = config.replace("ignore_eos: false", "ignore_eos: true")
        config = config.replace("min_tokens: 0", f"min_tokens: {length}")
        config = config.replace("max_completion_tokens: 1024", f"max_completion_tokens: {length}")
    (directory / "config.yaml").write_text(config)
    env = dict(os.environ, QUICK_ARTIFACTS=str(directory / "artifacts"))
    executable = str(Path.home() / "aiperf-venv/bin/aiperf")
    args = ["timeout", "1800", executable, "profile", "--config", str(directory / "config.yaml")]
    print(f"START {command['phase']}", flush=True)
    with (directory / "client.log").open("w") as log:
        subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    command_path.with_suffix(".done").write_text("0\n")
    print(f"DONE {command['phase']}", flush=True)


if __name__ == "__main__":
    main()
