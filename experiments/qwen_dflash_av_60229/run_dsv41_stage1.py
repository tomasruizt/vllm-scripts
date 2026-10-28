"""Short real-prompt AIPerf calibration, leaving the server alive for stage two."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from urllib.request import Request, urlopen


def main() -> None:
    root, port, model = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    template = Path(__file__).with_name("full-curve-sweep.yaml").read_text().split("\nsweep:")[0]
    template = template.replace("model: qwen", "model: deepseek").replace("8676", port)
    old_model = next(line.split("name: ", 1)[1] for line in template.splitlines() if "name: /data/" in line)
    tokenizer = "deepseek-ai/DeepSeek-V4.1-Flash-1017\n    revision: 1bae0e1088993b9b82de08f59de5be5a34e42170"
    template = template.replace(old_model, tokenizer).replace("enable_thinking: false", "thinking: false")
    for c in (1, 32, 128):
        run_cell(root, template, c, warmup=True)
        run_cell(root, template, c, warmup=False)
    flush(root, port)
    (root / "STAGE1_DONE").write_text("Inspect relative total costs before stage two.\n")
    print("STAGE1_DONE", flush=True)


def run_cell(root: Path, template: str, concurrency: int, warmup: bool) -> None:
    phase = f"{'warmup' if warmup else 'calibrate'}-c{concurrency}"
    directory = root / "cells" / phase
    directory.mkdir(parents=True)
    control = root / "control.tmp"
    control.write_text(json.dumps(dict(mode="calibrate", phase=phase, trace=not warmup)))
    control.replace(root / "control.json")
    config = template.replace("concurrency: 1\n", f"concurrency: {concurrency}\n")
    config = config.replace("duration: 30", f"duration: {5 if warmup else 30}")
    config_path = directory / "config.yaml"
    config_path.write_text(config)
    args = [str(Path.home() / "aiperf-venv/bin/aiperf"), "profile", "--config", str(config_path)]
    (directory / "command.txt").write_text(shlex.join(args) + "\n")
    env = dict(os.environ, QUICK_ARTIFACTS=str(directory / "artifacts"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_CACHE="/data/engine/hub_cache")
    print(f"START {phase}", flush=True)
    with (directory / "client.log").open("w") as log:
        subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
    summary = json.loads((directory / "artifacts/profile_export_aiperf.json").read_text())
    if summary["request_error_rate"]["avg"] != 0:
        raise RuntimeError(f"Request errors in {phase}")
    print(f"DONE {phase}", flush=True)


def flush(root: Path, port: str) -> None:
    control = root / "control.tmp"
    control.write_text(json.dumps(dict(mode="native", phase="stage1-flush")))
    control.replace(root / "control.json")
    data = dict(model="deepseek", messages=[dict(role="user", content="Hello")],
                max_tokens=1, temperature=0, chat_template_kwargs=dict(thinking=False))
    request = Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(data).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        response.read()


if __name__ == "__main__":
    main()
