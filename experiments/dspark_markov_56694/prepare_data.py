"""Resolve SPEED-Bench with NVIDIA's loader and export AIPerf's message schema."""

import json
import sys
from pathlib import Path

from speed_loader.speed import SPEEDBench, TURNS_PLACEHOLDER


def main() -> None:
    root = Path(sys.argv[1])
    for bucket in ("8k", "16k"):
        export(root, bucket)


def export(root: Path, bucket: str) -> None:
    import pandas as pd

    dataset = root / "data/speed" / f"throughput_{bucket}"
    source = SPEEDBench.prepare_data(root / "data" / bucket, config_name=str(dataset))
    rows = pd.read_parquet(source).to_dict(orient="records")
    with (root / f"throughput_{bucket}.jsonl").open("w") as stream:
        stream.writelines(json.dumps(record(row)) + "\n" for row in rows)
    print(f"{bucket}: {len(rows)} resolved prompts", flush=True)


def record(row: dict) -> dict:
    assert all(TURNS_PLACEHOLDER not in turn for turn in row["turns"])
    return {**row, "turns": list(row["turns"]),
            "messages": [dict(role="user", content=row["turns"][0])]}


if __name__ == "__main__":
    main()
