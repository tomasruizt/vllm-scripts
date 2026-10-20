"""Retain original benchmark metrics alongside per-startup memory measurements."""
import json
from pathlib import Path
import re
import sys


def main() -> None:
    root = Path(sys.argv[1])
    rows = [arm_row(path) for path in sorted(root.glob("*/s*-*/server.log"))]
    measurements = [dict(case=path.parents[2].name, arm=path.parents[1].name, cell=path.parent.name,
                         metrics=json.loads(path.read_text())) for path in sorted(root.glob("*/s*-*/c*/measurement.json"))]
    (root / "startup-metrics.json").write_text(json.dumps(rows) + "\n")
    (root / "benchmark-metrics.json").write_text(json.dumps(measurements) + "\n")
    for row in rows:
        print(json.dumps(row))
    print(f"Collected {len(rows)} startup attempts and {len(measurements)} measured cells")


def arm_row(path: Path) -> dict:
    arm = path.parent
    text = path.read_text()
    start = int((arm / "start-ns.txt").read_text())
    ready = arm / "ready-ns.txt"
    failure = arm / "failure.txt"
    return dict(case=arm.parent.name, arm=arm.name, startup_s=(int(ready.read_text())-start)/1e9 if ready.exists() else None,
                graph_memory_gib=number(text, r"CUDA graph pool memory: ([\d.]+) GiB \(actual\)"),
                estimated_graph_memory_gib=number(text, r"GiB \(actual\), ([\d.]+) GiB \(estimated\)"),
                kv_memory_gib=number(text, r"Available KV cache memory: ([\d.]+) GiB"),
                kv_tokens=number(text, r"GPU KV cache size: ([\d,]+) tokens"),
                capture_s=number(text, r"Graph capturing finished in ([\d.]+) secs"),
                failure=failure.read_text() if failure.exists() else None)


def number(text: str, pattern: str) -> float | None:
    matches = re.findall(pattern, text)
    return float(matches[-1].replace(",", "")) if matches else None


if __name__ == "__main__":
    main()
