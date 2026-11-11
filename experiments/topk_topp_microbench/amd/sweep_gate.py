"""Run the gate once per env setting (dev overrides of the port), in one GPU job.

Usage: python sweep_gate.py TAG "HIER_BINS=256 HIER_WARPS=16" "HIER_BINS=512" ...
"""

import os
import subprocess
import sys


CASES = ["8:0:0.9", "64:0:0.95:real", "1:0:0.9", "256:0:0.9", "256:10000:0.9", "8:1000:1", "2048:0:0.95:real"]


def main() -> None:
    tag, settings = sys.argv[1], sys.argv[2:]
    for i, setting in enumerate(settings):
        env = dict(os.environ, **dict(kv.split("=") for kv in setting.split()))
        print(f"SWEEP {i} {setting}", flush=True)
        cmd = ["python3", "gate.py", f"results/{tag}-{i}.jsonl", "--logits-file",
               "qwen36-verify-logits-t1.pt", "--impl", "triton-hier"]  # fmt: skip
        if tag.startswith("dbg"):
            cmd = ["python3", "debug_rounds.py", "--cases", *CASES]
        subprocess.run(cmd, env=env, check=False)


if __name__ == "__main__":
    main()
