"""Export c256 cost observations and AIPerf metrics using the DuckDB CLI."""

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
from statistics import median
import subprocess

import build_report
from fit_cost_curves import write_parquet
from summarize_cost_oracle import load_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--prefix", default="qwen-cost-c256")
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path("results").glob(f"{args.prefix}-[ab]-20261019/cells/*/artifacts/profile_export_aiperf.json"))
    rows = [cell_metrics(path) for path in paths]
    if not rows:
        raise SystemExit("No completed c256 cells")
    write_parquet(rows, args.output / "metrics.parquet")
    build_report.write_csv([{k: v for k, v in row.items() if not k.endswith("_json")} for row in rows], args.output / "metrics.csv")
    sql = Path(__file__).with_name("cost_c256.sql").read_text().replace("__OUTPUT__", str(args.output))
    sql = sql.replace("__PREFIX__", args.prefix)
    subprocess.run(["duckdb"], input=sql, text=True, check=True)
    if args.freeze:
        freeze_lookup(args.output, args.prefix, args.tp)
    if any(row["mode"] == "oracle" and not row["traced"] for row in rows):
        compare_policies(args.output, args.prefix)
    coverage = list(csv.DictReader((args.output / "coverage.csv").open()))
    print(json.dumps([row for row in coverage if row["fold"] == "pooled" and int(row["num_reqs"]) >= 254], indent=2))


def cell_metrics(path: Path) -> dict:
    row = load_metrics(path)
    phase = row["phase"].split("-")
    row.update(fold=re.search(r"-([ab])-20261019", row["suite"])[1], mode="sweep" if phase[0].startswith("burst") else phase[1],
               repeat=int(phase[0][1:]) if phase[0].startswith("r") else None, traced=phase[0] == "trace")
    return row


def compare_policies(output: Path, prefix: str) -> None:
    sql = Path(__file__).with_name("cost_oracle_policy.sql").read_text()
    sql = sql.replace("policy-([ab])-", "-([ab])-20261019")
    sql = sql.replace("results/qwen-av-cost-oracle-policy-*-20261019", f"results/{prefix}-[ab]-20261019")
    sql = sql.replace("WHERE phase LIKE 'trace-%'", "WHERE phase LIKE 'trace-%-post'")
    sql = sql.replace("results/cost-oracle-analysis-20261019/points-frozen.parquet", str(output / "points-frozen.parquet"))
    subprocess.run(["duckdb"], input=sql.replace("__OUTPUT__", str(output)), text=True, check=True)
    print((output / "comparison.csv").read_text())
    print((output / "decisions.csv").read_text())
    plot_comparison(output)


def plot_comparison(output: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/qwen-av-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = json.loads((output / "cost-lookup-frozen.json").read_text())["lookup"]["pooled/256"]
    measured = [row["mean_ms"] for row in lookup]
    query = f"SELECT fold, first(original_costs_ms ORDER BY sequence) AS costs FROM '{output}/decisions.parquet' WHERE steady AND eligible AND num_reqs=256 AND policy='native' GROUP BY fold ORDER BY fold"
    native = json.loads(subprocess.run(["duckdb", "-json", "-c", query], capture_output=True, text=True, check=True).stdout)
    pairs = list(csv.DictReader((output / "pairs.csv").open()))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), layout="constrained")
    for row in native:
        values = row["costs"]
        axes[0].plot(values, label=f"Native {row['fold']}", linestyle="--", linewidth=1)
        axes[1].plot([value / values[0] for value in values], linestyle="--", linewidth=1)
    axes[0].plot(measured, label="Measured mean", color="black", linewidth=.8)
    axes[1].plot([value / measured[0] for value in measured], color="black", linewidth=.8)
    axes[0].set(xlabel="Global verification budget", ylabel="Step cost (ms)", title="Absolute prices")
    axes[1].set(xlabel="Global verification budget", ylabel="Cost / budget-zero cost", title="Relative prices")
    axes[0].legend(frameon=False)
    for fold, offset in (("a", -.06), ("b", .06)):
        rows = [row for row in pairs if row["fold"] == fold]
        axes[2].scatter([int(row["repeat"]) + offset for row in rows], [float(row["gain_pct"]) for row in rows], label=f"Fold {fold}")
    if pairs:
        axes[2].axhline(median(float(row["gain_pct"]) for row in pairs), color="black", linewidth=1)
    axes[2].axhline(0, color="gray", linestyle="--", linewidth=.8)
    axes[2].set(xlabel="Paired repeat", ylabel="Throughput change (%)", xticks=[1, 2, 3], title="Measured costs vs native AV")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=.2)
    for extension in ("png", "svg"):
        fig.savefig(output / f"costs-and-gains.{extension}", dpi=150)
    plt.close(fig)


def freeze_lookup(output: Path, prefix: str, tp: int) -> None:
    path = output / "cost-lookup-frozen.json"
    if path.exists():
        raise SystemExit(f"Refusing to overwrite frozen lookup: {path}")
    ranks = list(csv.DictReader((output / "rank-agreement.csv").open()))
    if not ranks or any(int(row["disagreements"]) for row in ranks):
        raise SystemExit("TP rank agreement must pass before freezing")
    rows = list(csv.DictReader((output / "points.csv").open()))
    lookup = {}
    capacities = {}
    for row in rows:
        r = int(row["num_reqs"])
        capacity = int(row["candidate_capacity"])
        assert capacities.setdefault(r, capacity) == capacity
        key = f"{row['fold']}/{r}"
        lookup.setdefault(key, [None] * (capacity + 1))[int(row["budget"])] = dict(
            samples=int(row["samples"]), mean_ms=float(row["mean_ms"]), ms=float(row["median_ms"]))
    for r in (254, 255, 256):
        full = lookup.get(f"pooled/{r}", [])
        if not full or len(full) != capacities[r] + 1 or any(row is None or row["samples"] < 5 for row in full):
            raise SystemExit(f"R{r} needs every feasible budget with at least five samples before freezing")
    commits = sorted({p.read_text().strip() for p in Path("results").glob(f"{prefix}-[ab]-20261019/commit.txt")})
    data = dict(max_num_seqs=256, tensor_parallel_size=tp, drafter_k=7, minimum_samples=5, acceptance_estimation="unchanged",
                domain="complete directly observed decode budget vectors only, native fallback elsewhere",
                capacities=capacities,
                timing="batch t priced by gap t+1 with three equal-R prefill-free neighbors",
                source_commits=commits, points_sha256=sha256((output / "points.parquet").read_bytes()).hexdigest(), lookup=lookup)
    path.write_text(json.dumps(data, indent=2) + "\n")
    shutil.copyfile(output / "points.parquet", output / "points-frozen.parquet")
    print(f"Frozen lookup SHA256 {sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
