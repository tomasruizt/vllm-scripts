"""Resolve SPEED-Bench throughput_1k with NVIDIA's loader and export AIPerf's message schema."""

import json
import sys
from pathlib import Path

import pandas as pd
from speed_loader.speed import SPEEDBench, TURNS_PLACEHOLDER


def main() -> None:
    root = Path(sys.argv[1])
    source = SPEEDBench.prepare_data(root / "data/1k", config_name=str(root / "data/speed/throughput_1k"))
    rows = pd.read_parquet(source).to_dict(orient="records")
    with (root / "throughput_1k.jsonl").open("w") as stream:
        stream.writelines(json.dumps(record(row)) + "\n" for row in rows)
    print(f"throughput_1k: {len(rows)} resolved prompts", flush=True)


def record(row: dict) -> dict:
    assert all(TURNS_PLACEHOLDER not in turn for turn in row["turns"])
    return {**row, "turns": list(row["turns"]), "messages": [dict(role="user", content=row["turns"][0])]}


if __name__ == "__main__":
    main()
