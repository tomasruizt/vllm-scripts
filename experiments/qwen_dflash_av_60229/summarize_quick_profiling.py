"""Collect the three-arm serving comparison, preserving original AIPerf records."""

import argparse
from pathlib import Path

import build_report
from fit_cost_curves import write_parquet
from summarize_cost_oracle import load_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    paths = sorted(args.source.glob("on-*/repeat-1/artifacts-c*/profile_export_aiperf.json"))
    if not paths:
        raise SystemExit("No completed benchmark cells")
    rows = [load_cell(path, args.source) for path in paths]
    write_parquet(rows, args.source / "metrics.parquet")
    summary = [{key: value for key, value in row.items() if not key.endswith("_json")} for row in rows]
    build_report.write_csv(summary, args.source / "metrics.csv")
    print(f"Collected {len(rows)}/9 cells, {sum(row['valid'] for row in rows)} passed validity checks")
    for row in rows:
        print(f"{row['arm']} c{row['concurrency']}: valid={row['valid']} {row['failed_gates']}")


def load_cell(path: Path, source: Path) -> dict:
    row = load_metrics(path)
    row.update(arm=path.relative_to(source).parts[0].removeprefix("on-"),
               suite=source.name, phase="measurement", tensor_parallel_size=2,
               max_num_seqs=256, capture_grid="default", instrumentation="none",
               source_json=str(path.relative_to(source)))
    return row


if __name__ == "__main__":
    main()
