"""Build a per-run Parquet table from the benchmark result JSON files."""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis" / "runtimes.parquet"
RESULTS = ROOT / "results" / "qwen3_32b_fp8_tp2_complete"

METRICS = (
    "duration", "completed", "failed", "total_input_tokens",
    "total_output_tokens", "request_throughput", "output_throughput",
    "total_token_throughput", "mean_ttft_ms", "median_ttft_ms",
    "p99_ttft_ms", "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
    "mean_e2el_ms", "median_e2el_ms", "p99_e2el_ms",
    "spec_decode_acceptance_rate", "spec_decode_acceptance_length",
    "spec_decode_num_drafts", "spec_decode_draft_tokens",
    "spec_decode_accepted_tokens",
)


def main() -> None:
    rows = []
    for path in sorted(RESULTS.glob("SERVE--*-BENCH--c*/run=*.json")):
        match = re.fullmatch(r"SERVE--(.+)-BENCH--c(\d+)", path.parent.name)
        if match is None:
            continue
        data = json.loads(path.read_text())
        experiment = path.parent.parent.name
        repeat = int(path.stem.split("=")[1])
        arm = re.fullmatch(r"(fixed|adaptive)_k(\d+)", match[1])
        if arm is None:
            raise ValueError(f"Unexpected arm in {path}: {match[1]}")
        row = {
            "experiment": experiment,
            "mode": arm[1],
            "k": int(arm[2]),
            "concurrency": int(match[2]),
            "repeat": repeat,
            "result_path": str(path),
            **{name: data.get(name) for name in METRICS},
        }
        row["interactivity_tok_s_user"] = 1000 / row["median_tpot_ms"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUTPUT, index=False)
    print(f"Wrote {len(frame)} runs to {OUTPUT}")


if __name__ == "__main__":
    main()
