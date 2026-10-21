"""Export empirical lookup points and original AIPerf metrics, without curve fitting."""

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path

import build_report
from fit_cost_curves import write_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    points = list(csv.DictReader((args.output / "points.csv").open()))
    lookup = {}
    for row in points:
        fold = row["fold"]
        r = int(row["num_reqs"])
        key = f"{fold}/{r}"
        lookup.setdefault(key, [None] * (7 * r + 1))[int(row["budget"])] = dict(
            ms=float(row["median_ms"]), samples=int(row["samples"]),
            mean_ms=float(row["mean_ms"]), std_ms=float(row["std_ms"]) if row["std_ms"] else None,
            supported=int(row["samples"]) >= 10, probe_ms=float(row["probe_ms"]),
            context_mean_range=[float(row["min_context_mean"]), float(row["max_context_mean"])],
            adjusted_ms=float(row["adjusted_ms"]),
            p10_ms=float(row["p10_ms"]), p90_ms=float(row["p90_ms"]))
    commits = sorted({p.read_text().strip() for p in Path("results").glob("qwen-cost-oracle-*-20261019/commit.txt")})
    data = dict(method="direct observed medians, no fit, no interpolation, missing entries are null",
                drafter_k=7, acceptance_estimation="unchanged",
                timing="batch t uses raw step-start gap t+1, one-step pipeline lag, including instrumentation",
                primary_num_reqs=[1, 8, 32, 64, 128], primary_table="pooled", minimum_samples=10,
                validation="native steps predicted exclusively using the other fold, pooled table excluded",
                domain="decode only, full 7R candidates, three equal-R prefill-free neighbors",
                gap_filter_ms=[0, 100],
                source_commits=commits,
                collector_sha256=sha256(Path("av_cost_oracle.py").read_bytes()).hexdigest(),
                patch_sha256=sha256(Path("av-cost-oracle.patch").read_bytes()).hexdigest(),
                lookup=lookup)
    (args.output / "cost-lookup.json").write_text(json.dumps(data, indent=2) + "\n")
    paths = sorted(Path("results").glob("qwen-cost-oracle-*-20261019/cells/*/artifacts/profile_export_aiperf.json"))
    rows = [load_metrics(path) for path in paths]
    if rows:
        write_parquet(rows, args.output / "metrics.parquet")
        summary = [{k: v for k, v in row.items() if not k.endswith("_json")} for row in rows]
        build_report.write_csv(summary, args.output / "metrics.csv")
    plot(points, args.output)
    print((args.output / "coverage.csv").read_text())
    print((args.output / "validation.csv").read_text())


def load_metrics(path: Path, *, phase_name: str | None = None) -> dict:
    data = json.loads(path.read_text())
    server = json.loads(path.with_name("server_metrics_export.json").read_text())
    requests = [json.loads(line) for line in path.with_name("profile_export.jsonl").read_text().splitlines()]
    if phase_name is not None:
        if len(data["input_config"]["phases"]) > 1:
            phase_dir = path.parent / "phases" / phase_name
            data = json.loads((phase_dir / "profile_export_aiperf.json").read_text())
        requests = [r for r in requests if r["metadata"]["phase_name"] == phase_name]
    done = [row for row in requests if "output_sequence_length" in row.get("metrics", {})]
    config = next(p for p in data["input_config"]["phases"] if p["name"] == "profiling")
    source = path.parents[3]
    phase = path.parents[1].name
    metric = server["metrics"]
    row = dict(suite=source.name, phase=phase, concurrency=config["concurrency"],
               scope="fixed_length_cost_calibration" if phase.startswith("burst") else "natural_eos",
               requests_target=config["requests"], requests=len(done),
               cancelled=sum(r["metadata"].get("was_cancelled", False) for r in requests),
               error_rate=data["request_error_rate"]["avg"],
               preemptions=build_report.server_total(metric, "vllm:num_preemptions"),
               server_running_p50=build_report.server_stat(metric, "vllm:num_requests_running", "p50"),
               server_waiting_p50=build_report.server_stat(metric, "vllm:num_requests_waiting", "p50"),
               acceptance_length=1 + build_report.server_total(metric, "vllm:spec_decode_num_accepted_tokens")
               / build_report.server_total(metric, "vllm:spec_decode_num_drafts"),
               aiperf_throughput=data["output_token_throughput"]["avg"],
               window_start_ns=sorted(r["metadata"]["request_end_ns"] for r in done)[min(config["concurrency"], len(done)) - 1],
               window_end_ns=max(r["metadata"]["request_start_ns"] for r in done),
               cell_start_ns=min(r["metadata"]["request_start_ns"] for r in done),
               cell_end_ns=max(r["metadata"]["request_end_ns"] for r in done),
               mean_input_tokens=data["input_sequence_length"]["avg"],
               mean_output_tokens=data["output_sequence_length"]["avg"],
               cap_hit_pct=100 * sum(r["metrics"]["output_sequence_length"]["value"] >= 1024 for r in done) / len(done),
               prompt_set=sha256("\n".join(sorted(r["metadata"]["conversation_id"] for r in requests)).encode()).hexdigest()[:12],
               aiperf_summary_json=json.dumps(data), aiperf_server_metrics_json=json.dumps(server),
               aiperf_request_records_json=json.dumps(requests),
               aiperf_outputs_json=path.with_name("outputs.json").read_text(),
               **build_report.steady_window(done, config["concurrency"]))
    row.update(build_report.gates(row))
    fixed_length = data["input_config"]["endpoint"]["extra"].get("min_tokens", 0)
    fixed_lengths = fixed_length > 0 and all(r["metrics"]["output_sequence_length"]["value"] == fixed_length for r in done)
    complete = row["requests"] == row["requests_target"] and row["cancelled"] == 0
    clean = row["error_rate"] == 0 and row["preemptions"] == 0 and row["server_waiting_p50"] == 0
    row["cost_valid"] = row["valid"] or (phase.startswith("burst") and fixed_lengths and complete and clean)
    return row


def plot(points: list[dict], output: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/qwen-av-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(16, 3), layout="constrained")
    for axis, r in zip(axes, (1, 8, 32, 64, 128)):
        for fold in ("a", "b"):
            rows = [row for row in points if row["fold"] == fold and int(row["num_reqs"]) == r]
            axis.plot([int(row["budget"]) for row in rows], [float(row["median_ms"]) for row in rows],
                      linewidth=.8, alpha=.8, label=f"Fold {fold}")
        axis.set(title=f"R={r}", xlabel="Verified drafts", ylabel="Step ms")
        axis.grid(alpha=.2)
    axes[0].legend(frameon=False)
    for extension in ("png", "svg"):
        fig.savefig(output / f"measured-costs.{extension}", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
