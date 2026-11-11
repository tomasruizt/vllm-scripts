"""Per-step view of a vLLM torch-profiler trace (one rank): step period, GPU busy time,
GPU idle gaps, the top-k/top-p filter's kernels, and the host runtime calls by total time.

Usage: prof_steps.py TRACE.json[.gz] [--top 15]
"""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

from prof_kernels import is_filter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    opener = gzip.open if args.trace.suffix == ".gz" else open
    events = [e for e in json.load(opener(args.trace))["traceEvents"] if e.get("ph") == "X"]
    steps = sorted(e["ts"] for e in events if e.get("cat") == "user_annotation" and e["name"].startswith("execute_"))
    period = (steps[-1] - steps[0]) / (len(steps) - 1)
    lo, hi = steps[0], steps[-1]
    n = len(steps) - 1
    kernels = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and lo <= e["ts"] < hi]
    busy = union(kernels)
    print(f"{n} steps, period {period / 1000:.2f} ms, GPU busy {busy / n / 1000:.2f} ms/step, idle {(period - busy / n) / 1000:.2f} ms/step")
    print_group("GPU", kernels, n, args.top, lambda e: e["name"])
    filt = [e for e in kernels if is_filter(e["name"])]
    print(f"filter kernels: {len(filt) / n:.2f}/step, {sum(e['dur'] for e in filt) / n:.1f} us/step")
    runtime = [e for e in events if e.get("cat") == "cuda_runtime" and lo <= e["ts"] < hi]
    print_group("host runtime", runtime, n, args.top, lambda e: e["name"])
    cpu = [e for e in events if e.get("cat") in ("cpu_op", "user_annotation") and lo <= e["ts"] < hi]
    print_group("cpu ops", cpu, n, args.top, lambda e: e["name"])


def union(events: list[dict]) -> float:
    """Time covered by at least one interval."""
    total, end = 0.0, float("-inf")
    for e in sorted(events, key=lambda e: e["ts"]):
        start, stop = e["ts"], e["ts"] + e["dur"]
        if stop > end:
            total += stop - max(start, end)
            end = stop
    return total


def print_group(title: str, events: list[dict], steps: int, top: int, key) -> None:
    totals, counts = defaultdict(float), defaultdict(int)
    for e in events:
        totals[key(e)] += e["dur"]
        counts[key(e)] += 1
    print(f"-- {title}: us/step, calls/step, us/call")
    for name, total in sorted(totals.items(), key=lambda kv: -kv[1])[:top]:
        print(f"{total / steps:9.1f} {counts[name] / steps:6.2f} {total / counts[name]:8.1f}  {name[:100]}")


if __name__ == "__main__":
    main()
