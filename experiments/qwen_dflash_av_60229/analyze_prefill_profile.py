"""Per-step breakdown of a vLLM worker torch trace: wall period, CPU time, GPU kernel time, top kernels.

A step is the window from one `execute_context_*` annotation to the next. Kernels are assigned to the step
in which their CPU launch happened (via the correlation id), so async scheduling does not smear steps.
Usage: python analyze_prefill_profile.py <trace.json.gz> [--top 15]
"""

import argparse
import bisect
import gzip
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field

STEP_RE = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")


@dataclass
class Step:
    name: str
    ctx_tokens: int
    gen_tokens: int
    start: float
    exec_ms: float
    end: float = 0.0
    kernels: list[tuple[str, float, float]] = field(default_factory=list)  # (name, ts, dur) in us
    graph_launches: int = 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("trace")
    p.add_argument("--top", type=int, default=15)
    args = p.parse_args()
    steps = build_steps(load_events(args.trace))
    print_step_table(steps)
    for kind, pick in [("prefill", lambda s: s.ctx_tokens > 0), ("decode", lambda s: s.ctx_tokens == 0)]:
        chosen = [s for s in steps if pick(s)]
        if chosen:
            print_top_kernels(kind, chosen, args.top)


def load_events(path: str) -> list[dict]:
    with gzip.open(path) if path.endswith(".gz") else open(path) as f:
        return json.load(f)["traceEvents"]


def build_steps(events: list[dict]) -> list[Step]:
    steps = sorted((make_step(e) for e in events if is_step(e)), key=lambda s: s.start)
    for cur, nxt in zip(steps, steps[1:]):
        cur.end = nxt.start
    if steps:
        steps[-1].end = steps[-1].start + steps[-1].exec_ms * 1e3
    assign_kernels(events, steps)
    return steps[:-1]  # the last step's window is not closed


def is_step(e: dict) -> bool:
    return e.get("ph") == "X" and e.get("cat") == "user_annotation" and bool(STEP_RE.match(e.get("name", "")))


def make_step(e: dict) -> Step:
    m = STEP_RE.match(e["name"])
    assert m
    _, ctx_tok, _, gen_tok = map(int, m.groups())
    return Step(e["name"], ctx_tok, gen_tok, e["ts"], e["dur"] / 1e3)


def assign_kernels(events: list[dict], steps: list[Step]) -> None:
    launches = {e["args"]["correlation"]: e for e in events if is_launch(e)}
    starts = [s.start for s in steps]
    for e in events:
        if e.get("ph") == "X" and e.get("cat") == "kernel":
            assign_kernel(e, launches.get(e["args"].get("correlation")), steps, starts)
    for e in launches.values():
        step = find_step(steps, starts, e["ts"])
        if step and "GraphLaunch" in e["name"]:
            step.graph_launches += 1


def is_launch(e: dict) -> bool:
    return e.get("ph") == "X" and e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {})


def assign_kernel(kernel: dict, launch: dict | None, steps: list[Step], starts: list[float]) -> None:
    step = find_step(steps, starts, launch["ts"] if launch else kernel["ts"])
    if step:
        step.kernels.append((kernel["name"], kernel["ts"], kernel["dur"]))


def find_step(steps: list[Step], starts: list[float], ts: float) -> Step | None:
    i = bisect.bisect_right(starts, ts) - 1
    return steps[i] if i >= 0 and ts < steps[i].end else None


def print_step_table(steps: list[Step]) -> None:
    print(f"{'step':<44} {'period':>8} {'exec':>8} {'gpu_sum':>8} {'gpu_busy':>9} {'kernels':>8} {'graphs':>7}")
    for s in steps:
        period = (s.end - s.start) / 1e3
        gpu_sum = sum(d for _, _, d in s.kernels) / 1e3
        print(f"{s.name:<44} {period:8.2f} {s.exec_ms:8.2f} {gpu_sum:8.2f} {busy_ms(s.kernels):9.2f} {len(s.kernels):8d} {s.graph_launches:7d}")


def busy_ms(kernels: list[tuple[str, float, float]]) -> float:
    """Union of kernel intervals, so overlapping streams are not double counted."""
    total = 0.0
    cur_end = float("-inf")
    for _, ts, dur in sorted(kernels, key=lambda k: k[1]):
        total += max(0.0, ts + dur - max(ts, cur_end))
        cur_end = max(cur_end, ts + dur)
    return total / 1e3


def print_top_kernels(kind: str, steps: list[Step], top: int) -> None:
    per_name: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    for s in steps:
        for name, _, dur in s.kernels:
            per_name[name][0] += dur
            per_name[name][1] += 1
    total = sum(v[0] for v in per_name.values())
    print(f"\nTop kernels in {kind} steps (n={len(steps)}), ms per step, % of {kind} kernel time:")
    for name, (dur, cnt) in sorted(per_name.items(), key=lambda kv: -kv[1][0])[:top]:
        print(f"{dur / len(steps) / 1e3:8.2f} {100 * dur / total:5.1f}% {cnt / len(steps):6.0f}x  {name[:110]}")


if __name__ == "__main__":
    main()
