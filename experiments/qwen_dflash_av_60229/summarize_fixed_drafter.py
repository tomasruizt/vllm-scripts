"""Collect fixed-drafter calibration and fresh runtime controls into Parquet."""

import argparse
import json
from pathlib import Path

import build_report
from fit_cost_curves import trace_steps, write_parquet
from summarize_cost_experiment import budget_records, draw_summary, load_runs, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_runs(args.calibration, True) + load_runs(args.controls, False)
    for row in rows:
        if row["mode"] == "oncgdiag":
            row["arm"] = "B-traced"
    write_parquet(rows, args.output / "metrics.parquet")
    write_parquet(trace_steps(args.calibration) + trace_steps(args.controls), args.output / "steps.parquet")
    write_parquet(budget_records(args.calibration) + budget_records(args.controls), args.output / "budgets.parquet")
    summary = summarize(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    build_report.write_csv(summary, args.output / "summary.csv")
    primary = [row for row in summary if row["arm"] in ("B", "C", "D-k5")]
    build_report.write_csv(primary, args.output / "comparison.csv")
    draw_summary(primary, args.output)
    print(json.dumps(primary, indent=2))


if __name__ == "__main__":
    main()
