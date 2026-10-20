"""Combine serving repeats while retaining the independent server startup IDs."""
import json
from pathlib import Path
import re
import sys

from summarize import load_cell


def main() -> None:
    original = Path(sys.argv[1])
    repeated = Path(sys.argv[2])
    roots = [original, *sorted(repeated.glob("qwen38-recipe-startup-*"))]
    rows = [startup_cell(path, root) for root in roots for path in sorted(root.glob("*/artifacts/**/profile_export_aiperf.json"))]
    assert len(rows) == 90, len(rows)
    assert len({(row["variant"], row["startup"]) for row in rows}) == 9
    (repeated / "metrics.json").write_text(json.dumps(rows) + "\n")
    for row in rows:
        if not row["valid"]:
            print("INVALID", row["variant"], row["startup"], row["concurrency"], row["failed_gates"])
    print(f"Collected {len(rows)} measurements across 9 server startups, {sum(row['valid'] for row in rows)} valid")


def startup_cell(path: Path, root: Path) -> dict:
    row = load_cell(path, root)
    startup_path = root / "startup-id.txt"
    row["startup"] = int(startup_path.read_text()) if startup_path.exists() else 1
    log = (root / path.relative_to(root).parts[0] / "server.log").read_text()
    match = re.search(r"CUDA graph pool memory: [\d.]+ GiB \(actual\), ([\d.]+) GiB \(estimated\)", log)
    row["estimated_graph_memory_gib"] = float(match[1]) if match else None
    match = re.search(r"Available KV cache memory: ([\d.]+) GiB", log)
    row["kv_memory_gib"] = float(match[1]) if match else None
    return row


if __name__ == "__main__":
    main()
