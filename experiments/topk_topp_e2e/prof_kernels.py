"""GPU kernel time per engine step from a vLLM torch-profiler trace: the kernels with the
largest total time, and the top-k/top-p filter's share.

Usage: prof_kernels.py TRACE_DIR [--top 25]
"""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    trace = max(args.trace_dir.rglob("*.json*"), key=lambda p: p.stat().st_size)
    events = json.load(gzip.open(trace) if trace.suffix == ".gz" else open(trace))["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    steps = count_steps(events)
    totals, counts = defaultdict(float), defaultdict(int)
    for e in kernels:
        totals[e["name"]] += e["dur"]
        counts[e["name"]] += 1
    span = (max(e["ts"] + e["dur"] for e in kernels) - min(e["ts"] for e in kernels)) / 1000
    print(f"trace {trace.name}: {steps} steps, {span:.1f} ms window, {sum(totals.values()) / 1000 / steps:.2f} ms kernel time per step")
    filt = sum(v for k, v in totals.items() if is_filter(k))
    print(f"top-k/top-p filter: {filt / 1000 / steps:.3f} ms per step")
    for name, total in sorted(totals.items(), key=lambda kv: -kv[1])[: args.top]:
        print(f"{total / 1000 / steps:8.3f} ms/step {counts[name] / steps:6.1f} calls/step  {name[:110]}")


def count_steps(events: list[dict]) -> int:
    """Engine steps in the window: the V2 runner annotates each forward as
    execute_context_<n>(<tokens>)_generation_<n>(<tokens>). The sampler runs after
    that range, so per-step totals divide by the step count rather than use the range."""
    steps = sum(e.get("cat") == "user_annotation" and e["name"].startswith("execute_") for e in events)
    return max(steps, 1)


FILTER_KERNELS = ("_topk_topp_kernel", "_topp_sb", "_levels_kernel", "_filter_kernel")


def is_filter(name: str) -> bool:
    return name.startswith(FILTER_KERNELS)


if __name__ == "__main__":
    main()
