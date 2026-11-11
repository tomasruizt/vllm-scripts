"""Capture real processed logits using the existing server/AIPerf lifecycle."""

import argparse
import json
import signal
from pathlib import Path

import yaml

from run import DEFAULTS, Level, record_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    level = make_level(output)
    signal.signal(signal.SIGTERM, lambda *_: exit(1))
    try:
        level.start_server([], level.root / "server", level.config["serve_args"])
        for concurrency in (1, 32):
            workload = output / f"workload-c{concurrency}"
            workload.mkdir()
            capture = output / "captures" / f"c{concurrency}"
            level.rpc("arm_capture", [str(capture), 2 if concurrency == 1 else 64])
            client = level.aiperf(workload, "capture", duration=25, concurrency=concurrency)
            if client.wait(timeout=180) != 0:
                raise RuntimeError(f"AIPerf failed: {workload}")
            status = level.rpc("capture_status")
            (workload / "capture-status.json").write_text(json.dumps(status, indent=2))
            if not all(item["saved"] == 8 for item in status["results"]):
                raise RuntimeError(f"incomplete capture: {status}")
    finally:
        level.stop_server()
    (output / "DONE").write_text("ok\n")


def make_level(output: Path) -> Level:
    output.mkdir(parents=True, exist_ok=True)
    config = {**DEFAULTS, **yaml.safe_load(Path("config.yaml").read_text())}
    config["request_extra"] = {**config["request_extra"], "seed": 42}
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    record_source(config, output)
    level = Level(config, 32, output)
    level.root.mkdir(parents=True, exist_ok=True)
    directory = level.root / "server"
    directory.mkdir()
    return level


if __name__ == "__main__":
    main()
