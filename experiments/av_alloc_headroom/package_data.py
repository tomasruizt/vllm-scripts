"""Pack the traces and replay results into a shareable bundle (parquet + README).

Writes results/data-release/: one trace parquet per run and mode, the combined replay
results, the per-request residuals, the cost tables and serve commands. Only valid runs.
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build_report import RUNS  # noqa: E402
from replay import load_steps  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "results/data-release"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for run in RUNS:
        for mode_dir in sorted((HERE / "results" / run).glob("*/trace.jsonl")):
            pack_trace(run, mode_dir.parent)
    shutil.copy(HERE / "results/all_runs.parquet", OUT / "replay_results.parquet")
    shutil.copy(HERE / "results/all_residuals.parquet", OUT / "request_residuals.parquet")
    print(sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6, "MB")


def pack_trace(run: str, mode_dir: Path) -> None:
    dest = OUT / "traces" / run
    dest.mkdir(parents=True, exist_ok=True)
    steps = load_steps(mode_dir)
    steps = steps.drop(columns=["kind"]).explode(["req_ids", "scheduled", "admitted", "accepted", "p"])
    steps = steps.rename(columns={"req_ids": "req_id"}).reset_index(names="step")
    steps.to_parquet(dest / f"{mode_dir.name}.parquet", compression="zstd")
    costs = next(json.loads(line) for line in open(mode_dir / "trace.jsonl") if line.startswith('{"kind":"costs"'))
    (dest / f"{mode_dir.name}_cost_tables.json").write_text(json.dumps(costs))
    shutil.copy(mode_dir / "serve-command.txt", dest / f"{mode_dir.name}_serve_command.txt")


if __name__ == "__main__":
    main()
