"""Collect all cost-curve arms into one Parquet dataset with validity gates."""

import argparse
import json
from pathlib import Path
from statistics import median

import build_report
from fit_cost_curves import cell_windows, trace_steps, write_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "fixed", "corrected", "calibration"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--startup", type=Path, help="fresh untraced B controls, retain older B as B-prior")
    parser.add_argument("--matched", type=Path, help="primary C controls on the same GPU pair as B/D")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in (args.baseline, args.fixed, args.corrected, args.calibration):
        rows.extend(load_runs(source, source == args.calibration))
    if args.startup:
        for row in rows:
            if row["arm"] == "B":
                row["arm"] = "B-prior"
        rows.extend(load_runs(args.startup, False))
    if args.matched:
        for row in rows:
            if row["arm"] == "C":
                row["arm"] = "C-4/5"
        rows.extend(load_runs(args.matched, False))
    write_parquet(rows, args.output / "metrics.parquet")
    steps = trace_steps(args.calibration) + trace_steps(args.corrected)
    write_parquet(steps, args.output / "steps.parquet")
    budgets = budget_records(args.calibration) + budget_records(args.corrected)
    write_parquet(budgets, args.output / "budgets.parquet")
    summary = summarize(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    build_report.write_csv(summary, args.output / "summary.csv")
    primary = [r for r in summary if r["arm"] in ("A", "B", "C", "D-k3", "D-k5")]
    build_report.write_csv(primary, args.output / "comparison.csv")
    draw_summary(summary, args.output)
    print(json.dumps(summary, indent=2))


def budget_records(source: Path) -> list[dict]:
    result = []
    for path in sorted(source.glob("*/repeat-*/av-trace.jsonl")):
        windows = cell_windows(path.parent)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["kind"] != "budget":
                continue
            window = next((w for w in windows if w["start"] <= row["time_ns"] < w["end"]), None)
            if window is not None:
                result.append(dict(suite=source.name, mode=path.parts[-3], repeat=int(path.parts[-2].split("-")[-1]),
                                   concurrency=window["concurrency"], cell=window["cell"],
                                   steady=window["steady_start"] <= row["time_ns"] < window["steady_end"], **row))
    return result


def load_runs(source: Path, calibration: bool) -> list[dict]:
    statuses = list(source.glob("worker-gpu*-exit-code.txt"))
    if not statuses or any(p.read_text().strip() != "0" for p in statuses):
        raise ValueError(f"Incomplete or failed workers: {source}")
    rows = []
    for path in sorted(source.glob("*/repeat-*/artifacts*/**/profile_export_aiperf.json")):
        row = build_report.load_run(path, source)
        row["av_enabled"] = row["mode"].startswith("on")
        row["instrumentation"] = "step trace" if "diag" in row["mode"] else "none"
        row["arm"] = arm(row["mode"], calibration)
        row["server_env"] = (source / path.relative_to(source).parts[0] / path.relative_to(source).parts[1] / "server-env.txt").read_text().strip()
        rows.append(row)
    return rows


def arm(mode: str, calibration: bool) -> str:
    if calibration:
        return "calibration-" + mode
    if mode == "offcg":
        return "A"
    if mode == "oncg":
        return "B"
    if "k3" in mode:
        return "D-k3"
    if "k5" in mode:
        return "D-k5"
    if mode.startswith("on") and "diag" in mode:
        return "C-traced"
    if mode.startswith("on"):
        return "C"
    raise ValueError(f"Unrecognized arm: {mode}")


def summarize(rows: list[dict]) -> list[dict]:
    keys = sorted({(r["concurrency"], r["arm"]) for r in rows if r["cell"] == "main"})
    result = []
    for c, name in keys:
        all_rows = [r for r in rows if (r["concurrency"], r["arm"], r["cell"]) == (c, name, "main")]
        valid = [r for r in all_rows if r["valid"]]
        record = dict(concurrency=c, arm=name, valid_runs=len(valid), total_runs=len(all_rows))
        for metric in ("throughput", "mean_tpot_ms", "acceptance_length"):
            values = [r[metric] for r in valid]
            record[metric] = median(values) if values else None
            record[metric + "_min"] = min(values) if values else None
            record[metric + "_max"] = max(values) if values else None
        result.append(record)
    return result


def draw_summary(rows: list[dict], output: Path) -> None:
    plt = build_report.plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    labels = {"A": "AV off, k=7", "B": "AV, startup costs", "C": "AV, measured costs", "D-k3": "AV off, k=3", "D-k5": "AV off, k=5"}
    metrics = {"throughput": "Output tok/s", "mean_tpot_ms": "TPOT (ms)", "acceptance_length": "AL"}
    for axis, (metric, ylabel) in zip(axes, metrics.items()):
        for arm, label in labels.items():
            selected = [r for r in rows if r["arm"] == arm and r["valid_runs"]]
            if not selected:
                continue
            xs = [r["concurrency"] for r in selected]
            ys = [r[metric] for r in selected]
            axis.plot(xs, ys, marker="o", markersize=3, label=label)
            axis.fill_between(xs, [r[metric + "_min"] for r in selected], [r[metric + "_max"] for r in selected], alpha=.12)
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 8, 32, 64, 128], labels=["1", "8", "32", "64", "128"])
        axis.set(xlabel="Concurrency", ylabel=ylabel)
        axis.grid(alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=5, frameon=False)
    for ext in ("png", "svg"):
        fig.savefig(output / f"cost-experiment.{ext}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
