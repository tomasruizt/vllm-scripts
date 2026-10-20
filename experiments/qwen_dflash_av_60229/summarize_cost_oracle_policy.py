"""Compare cost-only AV with native AV, keeping diagnostic traces separate."""

import argparse
import csv
import os
from pathlib import Path
import re
from statistics import median
import subprocess

import build_report
from fit_cost_curves import write_parquet
from summarize_cost_oracle import load_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path("results").glob("qwen-av-cost-oracle-policy-*-20261019/cells/*/artifacts/profile_export_aiperf.json"))
    rows = [policy_metrics(path) for path in paths]
    if not rows:
        raise SystemExit("No completed cost-only policy cells")
    write_parquet(rows, args.output / "metrics.parquet")
    summary = [{k: v for k, v in row.items() if not k.endswith("_json")} for row in rows]
    build_report.write_csv(summary, args.output / "metrics.csv")
    sql = Path(__file__).with_name("cost_oracle_policy.sql").read_text()
    sql = sql.replace("__OUTPUT__", str(args.output).replace("'", "''"))
    traces = list(Path("results").glob("qwen-av-cost-oracle-policy-*-20261019/cost-steps.jsonl"))
    if not traces or not all(path.stat().st_size for path in traces):
        sql = sql.split("-- Decision traces", 1)[0]
    subprocess.run(["duckdb"], input=sql, text=True, check=True)
    plot_pairs(args.output)
    print((args.output / "comparison.csv").read_text())
    if (args.output / "decisions.csv").exists():
        print((args.output / "decisions.csv").read_text())


def plot_pairs(output: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/qwen-av-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader((output / "pairs.csv").open()))
    positions = {1: 0, 8: 1, 128: 2}
    fig, axis = plt.subplots(figsize=(7, 3), layout="constrained")
    for fold, offset in (("a", -.06), ("b", .06)):
        points = [row for row in rows if row["fold"] == fold]
        x = [positions[int(row["concurrency"])] + offset + .022 * (int(row["repeat"]) - 2) for row in points]
        axis.scatter(x, [float(row["gain_pct"]) for row in points], label=f"Fold {fold}", s=30)
    for concurrency, position in positions.items():
        values = [float(row["gain_pct"]) for row in rows if int(row["concurrency"]) == concurrency]
        if values:
            axis.hlines(median(values), position - .18, position + .18, color="black", linewidth=2)
    axis.axhline(0, color="gray", linestyle="--", linewidth=.8)
    axis.set(xticks=[0, 1, 2], xticklabels=["1", "8", "128"], xlabel="Concurrency",
             ylabel="Paired throughput change (%)", title="Measured decode costs vs native AV", xlim=(-.4, 2.4))
    axis.grid(axis="y", alpha=.2)
    axis.legend(frameon=False)
    for extension in ("png", "svg"):
        fig.savefig(output / f"paired-gains.{extension}", dpi=150)
    plt.close(fig)


def policy_metrics(path: Path) -> dict:
    row = load_metrics(path)
    phase = row["phase"].split("-")
    row.update(fold=re.search(r"policy-([ab])-", row["suite"])[1], mode=phase[1],
               repeat=int(phase[0][1:]) if phase[0].startswith("r") else None,
               traced=phase[0] == "trace")
    return row


if __name__ == "__main__":
    main()
