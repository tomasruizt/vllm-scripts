"""Collect individual native AIPerf sweep trials into one long-format Parquet."""

import argparse
from pathlib import Path

import build_report
from fit_cost_curves import write_parquet
from summarize_cost_oracle import load_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--complete", action="store_true", help="Require all 72 valid, matched trials")
    args = parser.parse_args()
    paths = sorted(args.source.glob("worker-*/on-*/artifacts/profile_runs/trial_*/*/profile_export_aiperf.json"))
    if not paths:
        raise SystemExit("No completed native sweep trials")
    rows = [load_trial(path, args.source) for path in paths]
    write_parquet(rows, args.source / "metrics.parquet")
    summary = [{key: value for key, value in row.items() if not key.endswith("_json")} for row in rows]
    build_report.write_csv(summary, args.source / "metrics.csv")
    print(f"Collected {len(rows)}/72 trials, {sum(row['valid'] for row in rows)} passed validity checks")
    for row in rows:
        if not row["valid"]:
            print(f"INVALID {row['worker']} {row['arm']} c{row['concurrency']} trial{row['trial']}: {row['failed_gates']}")
    if args.complete:
        validate_complete(rows)


def validate_complete(rows: list[dict]) -> None:
    keys = {(r["worker"], r["arm"], r["concurrency"], r["trial"]) for r in rows}
    expected = {(w, a, c, t) for w in ("worker-a", "worker-b") for a in ("main", "community", "gdn")
                for c in (1, 8, 32, 64, 128, 256) for t in (1, 2)}
    if keys != expected or len(rows) != 72 or not all(r["valid"] for r in rows):
        raise SystemExit("Sweep incomplete or invalid, inspect metrics.csv")
    for c in (1, 8, 32, 64, 128, 256):
        for t in (1, 2):
            prompts = {r["prompt_set"] for r in rows if r["concurrency"] == c and r["trial"] == t}
            if len(prompts) != 1:
                raise SystemExit(f"Unmatched prompts at c{c} trial{t}")


def load_trial(path: Path, source: Path) -> dict:
    row = load_metrics(path, phase_name="profiling")
    parts = path.relative_to(source).parts
    row.update(worker=parts[0], arm=parts[1].removeprefix("on-"),
               trial=int(path.parent.parent.name.removeprefix("trial_")), suite=source.name,
               phase="measurement", tensor_parallel_size=2, max_num_seqs=256,
               capture_grid="default", instrumentation="none", prefix_caching=False,
               source_json=str(path.relative_to(source)))
    return row


if __name__ == "__main__":
    main()
