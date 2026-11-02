"""Merge the report's run-level summaries into one long-format Parquet."""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def main() -> None:
    paths = sys.argv[1:] or sorted((HERE / "report/perf").glob("*/summary.csv"))
    frames = [pd.read_csv(path) for path in paths]
    table = pd.concat(frames, ignore_index=True)
    out = HERE / "results/perf.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    print(f"{out}: {len(table)} runs, {table.suite.nunique()} suites")


if __name__ == "__main__":
    main()
