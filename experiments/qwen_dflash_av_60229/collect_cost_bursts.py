"""Fill dense cost grids on reserved live servers with synchronized fixed-length waves."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from cost_oracle_queue import flush_trace, run_cell


def main() -> None:
    root = Path(sys.argv[1])
    port = sys.argv[2]
    queue_pid = int(sys.argv[3])
    template = Path(__file__).with_name("steady-speedbench.yaml").read_text()
    command_line = Path(f"/proc/{queue_pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    assert "cost_oracle_queue.py" in command_line and str(root) in command_line
    current_phase = pause_queue(root, queue_pid)
    print(f"WAIT {current_phase}", flush=True)
    try:
        wait_for_client(queue_pid)
        collect(root, port, template)
        flush_trace(root, port)
        skip_replaced_commands(root, current_phase)
        (root / "STOP").touch()
        print("DONE controlled collection", flush=True)
    finally:
        os.kill(queue_pid, signal.SIGCONT)


def pause_queue(root: Path, queue_pid: int) -> str:
    for _ in range(20):
        os.kill(queue_pid, signal.SIGSTOP)
        result = subprocess.run(["ps", "--ppid", str(queue_pid), "-o", "stat="], capture_output=True, text=True)
        if result.stdout.strip():
            return json.loads((root / "control.json").read_text())["phase"]
        os.kill(queue_pid, signal.SIGCONT)
        time.sleep(.1)
    raise RuntimeError("Queue has no active client to wait for")


def wait_for_client(queue_pid: int) -> None:
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        result = subprocess.run(["ps", "--ppid", str(queue_pid), "-o", "stat="], capture_output=True, text=True)
        states = result.stdout.split()
        if not states or all(state.startswith("Z") for state in states):
            return
        time.sleep(.5)
    raise TimeoutError("Current AIPerf client did not complete")


def collect(root: Path, port: str, template: str) -> None:
    commands = root / "burst-commands"
    commands.mkdir(exist_ok=True)
    for index in range(60):
        if (root / "BURST_STOP").exists():
            return
        command = dict(phase=f"burst{index:02}-c128", mode="sweep", concurrency=128,
                       requests=256, fixed_output_tokens=512)
        path = commands / f"{index:03}.json"
        path.write_text(json.dumps(command))
        if not path.with_suffix(".done").exists():
            run_cell(root, port, template, path)


def skip_replaced_commands(root: Path, current_phase: str) -> None:
    for path in (root / "commands").glob("*.json"):
        if path.with_suffix(".done").exists() or json.loads(path.read_text())["phase"] == current_phase:
            continue
        path.with_suffix(".done").write_text("skipped, replaced with synchronized cost calibration\n")


if __name__ == "__main__":
    main()
