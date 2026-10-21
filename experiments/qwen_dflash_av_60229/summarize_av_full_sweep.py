"""Export native AIPerf sweep runs and same-server policy comparisons."""

import json
from pathlib import Path
import subprocess

from fit_cost_curves import write_parquet


def main() -> None:
    root = Path("results/qwen-av-full-sweep-20261020")
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    if (root / "metrics.jsonl").exists():
        files = [root / "metrics.jsonl"]
        rows = []
    elif (root / "pilot-validation.json").exists():
        files = sorted(root.glob("main-*/metrics.jsonl"))
        rows = json.loads((root / "pilot-validation.json").read_text())
    else:
        files = sorted(root.glob("*/metrics.jsonl"))
        rows = []
    rows += [json.loads(line) for path in files for line in path.read_text().splitlines()]
    if not rows:
        raise SystemExit("No completed sweep stages yet")
    write_parquet(rows, output / "metrics.parquet")
    query = f"""
        COPY (SELECT mode, concurrency, count(*) runs, bool_and(valid AND window_s>=15) all_valid,
            min(window_s) min_window_s, median(throughput) throughput,
            min(throughput) min_throughput, max(throughput) max_throughput,
            median(mean_tpot_ms) tpot_ms, median(acceptance_length) acceptance_length
            FROM '{output}/metrics.parquet' WHERE included GROUP BY mode,concurrency)
            TO '{output}/summary.parquet' (FORMAT PARQUET);
        COPY (SELECT n.concurrency, n.runs native_runs, m.runs measured_runs,
            n.throughput native_tok_s, m.throughput measured_tok_s,
            100*(m.throughput/n.throughput-1) gain_pct,
            n.tpot_ms native_tpot_ms, m.tpot_ms measured_tpot_ms,
            n.acceptance_length native_al, m.acceptance_length measured_al,
            n.all_valid AND m.all_valid all_valid
            FROM '{output}/summary.parquet' n JOIN '{output}/summary.parquet' m USING(concurrency)
            WHERE n.mode='native' AND m.mode='measured' ORDER BY concurrency)
            TO '{output}/comparison.csv' (HEADER);
        SELECT * FROM read_csv_auto('{output}/comparison.csv');
        COPY (SELECT * FROM read_json_auto('{root}/audit-*-rank*.json',filename=true))
            TO '{output}/audits.parquet' (FORMAT PARQUET);
    """
    subprocess.run(["duckdb"], input=query, text=True, check=True)


if __name__ == "__main__":
    main()
