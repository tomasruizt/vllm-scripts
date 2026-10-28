"""Summarize buffered c1 stage timings and Nsight kernel evidence locally."""

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen_dflash_av_60229"))
from fit_cost_curves import write_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--complete", action="store_true")
    parser.add_argument("--modes", nargs="+", default=["baseline", "fixed", "adaptive"])
    args = parser.parse_args()
    rows = load_steps(args.source)
    if rows:
        write_parquet(rows, args.source / "steps.parquet")
    summaries = [
        summarize(group, mode, phase)
        for mode in args.modes
        for phase in ("trace", "events-only")
        if (
            group := [
                row
                for row in rows
                if row["mode"] == mode
                and row["phase"] == phase
                and row["rank"] == 0
                and not row.get("has_prefill", True)
                and "target_gpu_ms" in row
                and "step_gpu_ms" in row
                and row["sampled_tokens"] > 0
            ]
        )
    ]
    kernels = load_kernels(args.source)
    result = {
        "stage_timings": summaries,
        "kernel_evidence": kernels,
        "profile_inputs": profile_inputs(args.source),
    }
    result["gpu_clocks"] = clock_summary(args.source, result["profile_inputs"])
    (args.source / "profile-summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    if summaries:
        with (args.source / "profile-summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=summaries[0].keys())
            writer.writeheader()
            writer.writerows(summaries)
    print(json.dumps(result, indent=2))
    if args.complete:
        assert (args.source / "DONE").exists()
        assert (args.source / "exit-code.txt").read_text().strip() == "0"
        assert len(summaries) == 2 * len(args.modes), "Missing profile mode or phase"
        assert len(result["profile_inputs"]) == 2 * len(args.modes)
        assert (
            len(
                {
                    tuple(record["conversation_ids"])
                    for record in result["profile_inputs"]
                }
            )
            == 1
        ), "Profile prompts differ"
        assert all(
            record["completed"] == 8 and record["errors"] == 0
            for record in result["profile_inputs"]
        )
        assert all(
            row["gpu_query_len"] == row["target_tokens"]
            for row in rows
            if row.get("num_reqs") == 1
        )


def load_steps(source: Path) -> list[dict]:
    result = []
    for path in sorted(source.glob("profiles/*/steps-*-rank*.jsonl")):
        phase = path.stem.rsplit("-rank", 1)[0].removeprefix("steps-")
        for line in path.read_text().splitlines():
            row = json.loads(line)
            row.update(mode=path.parent.name, phase=phase)
            result.append(row)
    return result


def summarize(rows: list[dict], mode: str, phase: str) -> dict:
    emitted = sum(row["sampled_tokens"] for row in rows)
    result = {
        "mode": mode,
        "phase": phase,
        "decode_steps": len(rows),
        "sampled_tokens": emitted,
        "sampled_per_step": emitted / len(rows),
        "mean_query_len": mean(row["gpu_query_len"] for row in rows),
        "median_query_len": median(row["gpu_query_len"] for row in rows),
        "min_query_len": min(row["gpu_query_len"] for row in rows),
        "max_query_len": max(row["gpu_query_len"] for row in rows),
        "mean_context_tokens": mean(row["context_tokens"] for row in rows),
    }
    fields = [
        "step_gpu_ms",
        "target_gpu_ms",
        "sample_gpu_ms",
        "drafter_gpu_ms",
        "confidence_cpu_ms",
        "confidence_gpu_ms",
        "prepare_cpu_ms",
        "execute_cpu_ms",
        "sample_tokens_cpu_ms",
        "inter_step_cpu_gap_ms",
        "inter_step_gpu_gap_ms",
    ]
    for field in fields:
        values = [row[field] for row in rows if field in row]
        result[f"mean_{field}"] = mean(values) if values else None
    result["worker_gpu_ms_per_sampled_token"] = (
        sum(row["step_gpu_ms"] + row.get("inter_step_gpu_gap_ms", 0) for row in rows)
        / emitted
    )
    result["target_gpu_ms_per_sampled_token"] = (
        sum(row["target_gpu_ms"] for row in rows) / emitted
    )
    result["drafter_gpu_ms_per_sampled_token"] = (
        sum(row.get("drafter_gpu_ms", 0) for row in rows) / emitted
    )
    result["query_histogram"] = dict(
        sorted(Counter(row["gpu_query_len"] for row in rows).items())
    )
    return result


def load_kernels(source: Path) -> list[dict]:
    rows = []
    for path in sorted(source.glob("profiles/*/stats_cuda_gpu_kern_sum.csv")):
        for record in csv.DictReader(path.open()):
            if (
                "gdn_decode" in record["Name"]
                or "fused_sigmoid_gating_delta_rule_update_kernel" in record["Name"]
                or "fused_recurrent_gated_delta_rule_packed_decode_kernel"
                in record["Name"]
            ):
                rows.append(
                    {
                        "mode": path.parent.name,
                        "name": record["Name"],
                        "instances": int(record["Instances"]),
                        "total_ms": float(record["Total Time (ns)"]) / 1e6,
                        "mean_us": float(record["Avg (ns)"]) / 1000,
                    }
                )
    return rows


def profile_inputs(source: Path) -> list[dict]:
    rows = []
    for path in sorted(
        source.glob("profiles/*/*-artifacts/**/profile_export_aiperf.json")
    ):
        phase = next(
            part for part in path.parts if part.endswith("-artifacts")
        ).removesuffix("-artifacts")
        if phase == "warmup":
            continue
        data = json.loads(path.read_text())
        requests = [
            json.loads(line)
            for line in path.with_name("profile_export.jsonl").read_text().splitlines()
        ]
        ids = [
            row["metadata"]["conversation_id"]
            for row in sorted(
                requests, key=lambda row: row["metadata"]["request_start_ns"]
            )
        ]
        rows.append(
            {
                "mode": path.relative_to(source).parts[1],
                "phase": phase,
                "completed": len(requests),
                "errors": data["request_error_rate"]["avg"],
                "aiperf_mean_tpot_ms": data["inter_token_latency"]["avg"],
                "conversation_ids": ids,
                "request_start_ns": min(
                    row["metadata"]["request_start_ns"] for row in requests
                ),
                "request_end_ns": max(
                    row["metadata"]["request_end_ns"] for row in requests
                ),
            }
        )
    return rows


def clock_summary(source: Path, inputs: list[dict]) -> list[dict]:
    path = source / "gpu-clocks.csv"
    if not path.exists():
        path = source.parent / "gpu-clocks.csv"
    if not path.exists():
        return []
    samples = []
    for row in csv.reader(path.open(), skipinitialspace=True):
        timestamp = datetime.strptime(row[0], "%Y/%m/%d %H:%M:%S.%f").replace(
            tzinfo=UTC
        )
        try:
            sm_mhz = float(row[3])
        except ValueError:
            sm_mhz = None
        samples.append((timestamp.timestamp() * 1e9, int(row[1]), sm_mhz))
    result = []
    for run in inputs:
        for gpu in sorted({sample[1] for sample in samples}):
            clocks = [
                sm
                for time_ns, index, sm in samples
                if index == gpu
                and run["request_start_ns"] <= time_ns <= run["request_end_ns"]
                and sm is not None
            ]
            if clocks:
                result.append(
                    {
                        "mode": run["mode"],
                        "phase": run["phase"],
                        "gpu": gpu,
                        "sm_mhz_mean": mean(clocks),
                        "sm_mhz_min": min(clocks),
                        "sm_mhz_max": max(clocks),
                        "samples": len(clocks),
                        "unavailable_samples": sum(
                            sm is None
                            for time_ns, index, sm in samples
                            if index == gpu
                            and run["request_start_ns"]
                            <= time_ns
                            <= run["request_end_ns"]
                        ),
                    }
                )
    return result


if __name__ == "__main__":
    main()
