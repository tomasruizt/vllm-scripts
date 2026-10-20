"""Collect exact-shape timings and profiler kernels for direct DuckDB queries."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    shapes = json.loads((args.results / "shapes.json").read_text())
    timings = [dict(**{k: v for k, v in shape.items() if k != "target_ms"},
                    scope="target_forward", repeat=repeat, gpu_ms=ms)
               for shape in shapes for repeat, ms in enumerate(shape["target_ms"], 1)]
    controls = json.loads((args.results / "gdn-state-slots.json").read_text())
    timings.extend(dict(**{k: v for k, v in control.items() if k != "kernel_ms"},
                        scope="gdn_kernel", repeat=repeat, gpu_ms=ms)
                   for control in controls for repeat, ms in enumerate(control["kernel_ms"], 1))
    startup = json.loads((args.results / "startup.json").read_text())
    timings.extend(dict(scope="startup_target", gpu_ms=sample["forward_ms"],
                        requests=sample["num_reqs"], query=sample["num_target_tokens"] / sample["num_reqs"],
                        repeat=index // 4 + 1, **sample)
                   for index, sample in enumerate(startup))
    kernels = [row for shape in shapes for row in kernel_rows(args.results, shape)]
    (args.results / "timings.json").write_text(json.dumps(timings))
    (args.results / "kernels.json").write_text(json.dumps(kernels))
    print(f"{len(timings)} measurements, {len(kernels)} kernel groups")


def kernel_rows(root: Path, shape: dict) -> list[dict]:
    trace = json.loads((root / f"{shape['label']}-trace.json").read_text())
    grouped = defaultdict(list)
    for event in trace["traceEvents"]:
        if event.get("cat") != "kernel":
            continue
        dimensions = {k: v for k, v in event.get("args", {}).items()
                      if k in ("grid", "block", "registers per thread", "shared memory")}
        key = (event["name"], json.dumps(dimensions, sort_keys=True))
        grouped[key].append(event["dur"])
    return [dict(label=shape["label"], requests=shape["requests"], query=shape["query"],
                 valid_states=shape["valid_states"], kernel=name, dimensions=dimensions,
                 count=len(durations), total_us=sum(durations),
                 per_forward_us=sum(durations) / 3)
            for (name, dimensions), durations in grouped.items()]


if __name__ == "__main__":
    main()
