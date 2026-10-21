"""Export full-curve benchmark metrics and compare the existing GPU-matched baseline."""

import hashlib
import json
from pathlib import Path
import subprocess

from fit_cost_curves import write_parquet
from summarize_cost_oracle import load_metrics


def main() -> None:
    root = Path("results/qwen-full-curve-c256-20261020")
    output = Path("results/full-curve-analysis-20261020")
    output.mkdir(exist_ok=True)
    paths = sorted(root.glob("cells/measured-*-c256/artifacts/profile_export_aiperf.json"))
    baseline = Path("results/qwen-cost-c256-tp2-a-20261019")
    paths += sorted(baseline.glob("cells/r*-native-c256/artifacts/profile_export_aiperf.json"))
    rows = [dict(load_metrics(path), mode="measured" if path.is_relative_to(root) else "native") for path in paths]
    write_parquet(rows, output / "metrics.parquet")
    digest = hashlib.sha256((root / "measured-curves-frozen.json").read_bytes()).hexdigest()
    audits = [dict(json.loads(path.read_text()), phase=path.stem.removeprefix("audit-"))
              for path in sorted(root.glob("audit-measured-*-c256-rank*.json"))]
    if len(audits) != 14 or any(row.get("measured_decisions") != row.get("decisions")
                              or row.get("sha256") != digest for row in audits):
        raise RuntimeError("Missing or invalid measured-only coverage audit")
    write_parquet(audits, output / "audits.parquet")
    query = f"""
        COPY (SELECT mode, count(*) runs, bool_and(valid) all_valid,
            median(throughput) throughput, min(throughput) throughput_min, max(throughput) throughput_max,
            median(mean_tpot_ms) tpot_ms, median(acceptance_length) acceptance_length,
            median(aiperf_throughput) aiperf_throughput
            FROM '{output}/metrics.parquet' WHERE phase NOT LIKE '%trace%' GROUP BY mode)
            TO '{output}/comparison.csv' (HEADER);
        COPY (SELECT * EXCLUDE(aiperf_summary_json, aiperf_server_metrics_json,
            aiperf_request_records_json, aiperf_outputs_json) FROM '{output}/metrics.parquet')
            TO '{output}/metrics.csv' (HEADER);
        SELECT mode, phase, valid, failed_gates, throughput, mean_tpot_ms, acceptance_length
            FROM '{output}/metrics.parquet' ORDER BY mode,phase;
        SELECT * FROM read_csv_auto('{output}/comparison.csv');
        SELECT 100*(m.throughput/n.throughput-1) gain_pct
            FROM read_csv_auto('{output}/comparison.csv') m,
                 read_csv_auto('{output}/comparison.csv') n
            WHERE m.mode='measured' AND n.mode='native';
        SELECT sum(decisions) decisions, sum(measured_decisions) measured,
            sum(mixed_decisions) mixed, sum(fallback_decisions) fallback
            FROM '{output}/audits.parquet' WHERE phase LIKE '%rank0';
        COPY (SELECT * FROM read_json_auto('{root}/timings-rank*.jsonl', filename=true))
            TO '{output}/timings.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);
        COPY (SELECT * FROM read_json_auto('{root}/decisions-rank*.jsonl', filename=true))
            TO '{output}/decisions.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    subprocess.run(["duckdb"], input=query, text=True, check=True)


if __name__ == "__main__":
    main()
