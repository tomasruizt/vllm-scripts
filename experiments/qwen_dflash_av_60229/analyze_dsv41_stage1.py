"""Compare native and serving cost ratios without changing acceptance estimates."""

from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def main() -> None:
    root = Path(sys.argv[1])
    rows = read_rows(root / "timings-rank0.jsonl")
    groups = {name: defaultdict(list) for name in ("draft", "verify")}
    for row in rows:
        if not row["phase"].startswith("calibrate-"):
            continue
        groups["verify"][row["padded_tokens"]].append(row["forward_ms"])
        if row["cg_mode"].endswith("FULL"):
            groups["draft"][row["num_reqs"]].append(row["drafter_ms"])
    points = {name: [(n, float(np.median(ms)), len(ms)) for n, ms in sorted(group.items()) if len(ms) >= 3]
              for name, group in groups.items()}
    if any(len(group) < 2 for group in points.values()):
        raise RuntimeError(f"Insufficient calibration coverage: {points}")
    curves = {f"{name}_curve": [[n, ms] for n, ms, count in group] for name, group in points.items()}
    (root / "stage1-serving-medians.json").write_text(json.dumps(dict(**curves, points=points), indent=2))
    draft_x, draft_y, _ = np.asarray(points["draft"]).T
    verify_x, verify_y, _ = np.asarray(points["verify"]).T
    draft_y, verify_y = np.maximum.accumulate(draft_y), np.maximum.accumulate(verify_y)
    decisions = []
    for row in read_rows(root / "decisions-rank0.jsonl"):
        if not row["phase"].startswith("calibrate-"):
            continue
        r, start, capacity = row["requests"], row["non_draft_tokens"], row["capacity"]
        if not (draft_x[0] <= r <= draft_x[-1] and verify_x[0] <= start and start + capacity <= verify_x[-1]):
            continue
        # Graph-bucket lookup, matching the native target-table construction.
        indices = np.searchsorted(verify_x, np.arange(start, start + capacity + 1))
        measured = np.interp(r, draft_x, draft_y) + verify_y[indices]
        native, expected = np.asarray(row["native_costs"]), np.asarray(row["expected_tokens"])
        selected = int(np.argmax(expected / measured))
        native_selected = int(np.argmax(expected / native))
        relative_error = (measured / measured[0]) / (native / native[0]) - 1
        decisions.append(dict(phase=row["phase"], requests=r, non_draft_tokens=start, capacity=capacity,
                              native_budget=native_selected, measured_budget=selected,
                              changed=selected != native_selected,
                              max_relative_price_difference=float(np.max(np.abs(relative_error)))))
    (root / "stage1-counterfactual-decisions.json").write_text(json.dumps(decisions))
    summary = dict(timing_rows=len(rows), measured_knots={k: len(v) for k, v in points.items()},
                   covered_traced_decisions=len(decisions),
                   changed_decision_fraction=float(np.mean([r["changed"] for r in decisions])) if decisions else None,
                   p90_relative_price_difference=float(np.quantile([r["max_relative_price_difference"] for r in decisions], .9)) if decisions else None,
                   limitations="Sparse stage-one interpolation, context/composition averaged. Counterfactual choices reuse the exact native acceptance estimates. Not a performance result or full-domain cost oracle.")
    (root / "stage1-summary.json").write_text(json.dumps(summary, indent=2))
    source = str(root / "timings-rank0.jsonl").replace("'", "''")
    target = str(root / "timings.parquet").replace("'", "''")
    subprocess.run(["duckdb", "-c", f"COPY (SELECT * FROM read_json_auto('{source}')) TO '{target}' (FORMAT PARQUET);"], check=True)
    print(json.dumps(summary, indent=2))


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


if __name__ == "__main__":
    main()
