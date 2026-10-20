"""Summarize py-spy raw (collapsed) stacks: where the engine core spends CPU time inside model execution.

Usage: python analyze_pyspy.py pyspy-prefill.txt [--within execute_model] [--top 30]
"""

import argparse
import re
from collections import Counter


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stacks")
    p.add_argument("--within", default="execute_model", help="keep only samples with a frame containing this")
    p.add_argument("--top", type=int, default=30)
    args = p.parse_args()
    all_stacks = load(args.stacks)
    stacks = [(f, n) for f, n in all_stacks if any(args.within in x for x in f)]
    total = sum(n for _, n in stacks)
    print(f"samples: {sum(n for _, n in all_stacks)} total, {total} within '{args.within}'")
    print_counter("inclusive, by frame", inclusive(stacks), total, args.top)
    print_counter("self, by deepest Python frame (+ native leaf)", by_python_leaf(stacks), total, args.top)


def load(path: str) -> list[tuple[list[str], int]]:
    rows = []
    for line in open(path):
        stack, _, count = line.rstrip().rpartition(" ")
        if stack:
            rows.append((stack.split(";"), int(count)))
    return rows


def inclusive(stacks: list[tuple[list[str], int]]) -> Counter:
    c: Counter = Counter()
    for frames, n in stacks:
        for f in set(map(short, frames)):
            c[f] += n
    return c


def by_python_leaf(stacks: list[tuple[list[str], int]]) -> Counter:
    c: Counter = Counter()
    for frames, n in stacks:
        py = [f for f in frames if ".py:" in f]
        native_leaf = frames[-1] if frames[-1] not in py else ""
        c[f"{short(py[-1]) if py else '?'}  ->  {short(native_leaf)[:50]}"] += n
    return c


def short(frame: str) -> str:
    """'fn (/long/path/pkg/mod.py:123)' -> 'fn (pkg/mod.py:123)'"""
    return re.sub(r"\(.*?/([^/]+/[^/]+\.py:\d+)\)", r"(\1)", frame)


def print_counter(title: str, c: Counter, total: int, top: int) -> None:
    print(f"\n{title}:")
    for k, v in c.most_common(top):
        print(f"  {100 * v / total:5.1f}%  {k[:150]}")


if __name__ == "__main__":
    main()
