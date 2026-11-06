"""The distinct step shapes of an nsys run's steady steps, for the ncu_fill pass to replay.

Usage: fill_shapes.py report/<RUN_ID>/c1/steps.parquet results/<RUN_ID>/c1/ncu-fill-shapes.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd

MAX_SHAPES = 16  # a kernel's variant follows the padded token count, which takes few distinct values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("steps", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-shapes", type=int, default=MAX_SHAPES, help="keep only the most frequent shapes (ncu replays cost minutes per kernel-thousand)")
    args = parser.parse_args()
    shapes = fill_shapes(pd.read_parquet(args.steps))[: args.max_shapes]
    args.output.write_text(json.dumps(shapes, indent=1))
    print(f"{len(shapes)} shapes -> {args.output}")


def fill_shapes(steps: pd.DataFrame) -> list[dict]:
    """One entry per (requests, padded tokens) of the steady steps, most frequent first, with the median executed
    token count and context per request of its steps."""
    steady = steps[steps.steady & steps.actual_tokens.notna()]
    groups = steady.groupby(["reqs", "padded_tokens"])
    rows = [shape(key, group) for key, group in groups]
    return sorted(rows, key=lambda row: -row["steps"])[:MAX_SHAPES]


def shape(key: tuple, group: pd.DataFrame) -> dict:
    reqs, padded = key
    context = (group.context / group.reqs).median()
    return {"reqs": int(reqs), "padded": int(padded), "tokens": int(group.actual_tokens.median()),
            "context": int(context), "steps": len(group)}


if __name__ == "__main__":
    main()
