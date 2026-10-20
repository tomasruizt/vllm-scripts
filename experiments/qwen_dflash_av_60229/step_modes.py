"""Test whether slow ITL steps are exactly the steps that contain a prefill.

Rebuilds the engine step timeline from client chunk arrival times: requests in the same step
receive their chunk within a few ms of each other, so arrivals are clustered into steps.
A step that contains a new request's prefill emits that request's first token, so each step
is labeled `has_prefill` if any first token lands in it. Then step duration is compared by label.
"""

import argparse
import json
from pathlib import Path

import numpy as np

GAP_MS = 4.0  # arrivals closer than this belong to the same step
SLOW_MS = 60.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    print("mode,c,repeat,steps,prefill_steps,slow_steps,slow_with_prefill,fast_with_prefill,"
          "p_slow_given_prefill,p_slow_given_decode_only,median_ms_prefill,median_ms_decode_only")
    for path in sorted(args.source.glob("*/repeat-*/artifacts-c*/profile_export.jsonl")):
        if not path.parent.name.endswith("-end"):
            print(cell_summary(path))


def cell_summary(path: Path) -> str:
    mode, repeat, cell = path.parts[-4], path.parts[-3].removeprefix("repeat-"), path.parent.name
    c = int(cell.removeprefix("artifacts-c"))
    times, is_first = arrivals(path)
    duration, has_prefill = steps(times, is_first)
    slow = duration > SLOW_MS
    stats = [len(duration), has_prefill.sum(), slow.sum(), (slow & has_prefill).sum(), (~slow & has_prefill).sum(),
             slow[has_prefill].mean(), slow[~has_prefill].mean(),
             np.median(duration[has_prefill]), np.median(duration[~has_prefill])]
    return ",".join([mode, str(c), repeat] + [f"{s:.3f}" if isinstance(s, float) else str(s) for s in stats])


def arrivals(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Absolute arrival time (ms) of every streamed chunk, and whether it is a request's first token."""
    times, firsts = [], []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        metrics = record["metrics"]
        if "time_to_first_token" not in metrics:
            continue
        first = record["metadata"]["request_start_ns"] / 1e6 + metrics["time_to_first_token"]["value"]
        chunk = first + np.cumsum([0.0, *metrics.get("inter_chunk_latency", {}).get("value", [])])
        times.append(chunk)
        firsts.append(np.arange(len(chunk)) == 0)
    order = np.argsort(np.concatenate(times))
    return np.concatenate(times)[order], np.concatenate(firsts)[order]


def steps(times: np.ndarray, is_first: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per step (after the first): time since the previous step, and whether it emitted a first token."""
    boundary = np.diff(times) > GAP_MS
    step_id = np.concatenate(([0], np.cumsum(boundary)))
    start = times[np.concatenate(([0], np.flatnonzero(boundary) + 1))]
    has_first = np.bincount(step_id, weights=is_first).astype(bool)
    return np.diff(start), has_first[1:]


if __name__ == "__main__":
    main()
