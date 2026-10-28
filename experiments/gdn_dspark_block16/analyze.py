"""Audit execution and correctness, and reuse the existing AIPerf collector."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen_dflash_av_60229"))

import build_report
from fit_cost_curves import write_parquet
from summarize_cost_oracle import load_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--extra-source", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--complete", action="store_true")
    parser.add_argument("--include-baseline", action="store_true")
    args = parser.parse_args()
    output = args.output or args.source
    output.mkdir(parents=True, exist_ok=True)
    sources = [args.source, *args.extra_source]
    rows = [row for source in sources for row in collect_metrics(source)]
    if args.extra_source:
        write_parquet(rows, output / "metrics.parquet")
        build_report.write_csv(
            [
                {key: value for key, value in row.items() if not key.endswith("_json")}
                for row in rows
            ],
            output / "metrics.csv",
        )
    summary = {
        "execution": execution(args.source),
        "correctness": correctness(args.source),
        "performance_cells": len(rows),
        "valid_performance_cells": sum(row["valid"] for row in rows),
    }
    (output / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if args.complete:
        for source in sources:
            assert (source / "DONE").exists(), f"Runner has not completed: {source}"
            assert (source / "exit-code.txt").read_text().strip() == "0", (
                f"Runner failed: {source}"
            )
        modes = (
            ("baseline", "fixed", "adaptive")
            if args.extra_source or args.include_baseline
            else ("fixed", "adaptive")
        )
        expected = {
            (mode, concurrency, trial)
            for mode in modes
            for concurrency in (1, 16, 64, 128)
            for trial in (1, 2, 3)
        }
        actual = {(row["mode"], row["concurrency"], row["trial"]) for row in rows}
        assert actual == expected and len(rows) == len(expected), (
            "Missing benchmark cells"
        )
        assert all(row["valid"] for row in rows), "Invalid benchmark cells"
        assert all(
            summary["execution"][key]
            for key in (
                "confidence_head",
                "confidence_consumed",
                "ragged_full_replay",
                "wide_full_replay",
                "width16_fused_disabled",
                "triton_fallback_kernel_instances",
                "mixed_prefill_decode_steps",
            )
        ), "Missing execution evidence"
        assert len(summary["correctness"]["scores"]) == 3, "Missing correctness arm"
        assert all(
            row["num_questions"] == 100
            for row in summary["correctness"]["scores"].values()
        )


def collect_metrics(source: Path) -> list[dict]:
    paths = sorted(
        source.glob("*/artifacts/profile_runs/**/profile_export_aiperf.json")
    )
    rows = [load_cell(path, source) for path in paths]
    if rows:
        write_parquet(rows, source / "metrics.parquet")
        build_report.write_csv(
            [
                {key: value for key, value in row.items() if not key.endswith("_json")}
                for row in rows
            ],
            source / "metrics.csv",
        )
    return rows


def load_cell(path: Path, source: Path) -> dict:
    row = load_metrics(path, phase_name="profiling")
    trial = (
        path.parent.parent.name
        if path.parent.parent.name.startswith("trial_")
        else path.parent.name
    )
    row.update(
        mode=path.relative_to(source).parts[0],
        trial=int(trial.removeprefix("trial_").removeprefix("run_")),
        source_json=str(path.relative_to(source)),
        suite=source.name,
        instrumentation="none",
        tensor_parallel_size=2,
        prefix_caching=False,
    )
    return row


def execution(source: Path) -> dict:
    path = source / "diagnostic/execution.jsonl"
    events = (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
    heads = [event for event in events if event["event"] == "confidence_head_loaded"]
    configurations = [
        event for event in events if event["event"] == "draft_configuration"
    ]
    events = [event for event in events if event.get("rank", 0) == 0]
    replays = [
        event
        for event in events
        if event["event"] == "target_full_graph_replayed"
        and event.get("query_lens")
        and event["descriptor"].get("max_query_len") == 16
    ]
    for event in replays:
        event["query_lens"] = [length for length in event["query_lens"] if length > 0]
    ragged = [event for event in replays if len(set(event["query_lens"])) > 1]
    wide = [event for event in replays if max(event["query_lens"]) > 8]
    mixed = [
        event
        for event in events
        if event["event"] == "gpu_query_lens"
        and any(0 < length <= 16 for length in event["query_lens"])
        and any(length > 16 for length in event["query_lens"])
    ]
    result = {
        "confidence_head": bool(heads)
        and all(event["enabled"] and event["present"] for event in heads),
        "confidence_head_ranks": sorted({event.get("rank", 0) for event in heads}),
        "draft_configurations": configurations,
        "confidence_consumed": any(
            event["event"] == "confidence_consumed" for event in events
        ),
        "width16_fused_disabled": any(
            event["event"] == "fused_gdn_eligibility"
            and event["state_width"] == 16
            and not event["eligible"]
            for event in events
        ),
        "full_replays_with_gpu_layout": len(replays),
        "ragged_full_replay": len(ragged),
        "wide_full_replay": len(wide),
        "mixed_prefill_decode_steps": len(mixed),
        "max_replayed_query_len": max(
            (max(event["query_lens"]) for event in replays), default=0
        ),
    }
    kernels = [
        row
        for path in (source / "diagnostic").glob("kernels*cuda_gpu_kern_sum.csv")
        for row in csv.DictReader(path.open())
    ]
    fallback = [
        row
        for row in kernels
        if "fused_sigmoid_gating_delta_rule_update_kernel" in row.get("Name", "")
    ]
    result["triton_fallback_kernel_instances"] = sum(
        int(row["Instances"]) for row in fallback
    )
    for key, matches in (("ragged_example", ragged), ("wide_example", wide)):
        result[key] = matches[0] if matches else None
    return result


def correctness(source: Path) -> dict:
    arms = {
        mode: json.loads(path.read_text())
        for mode in ("baseline", "fixed", "adaptive")
        if (path := source / mode / "gsm8k.json").exists()
    }
    result = {
        "scores": {mode: data["score"] for mode, data in arms.items()},
        "comparisons": {},
    }
    if "baseline" not in arms:
        return result
    baseline = arms["baseline"]["records"]
    for mode, data in arms.items():
        if mode == "baseline":
            continue
        records = data["records"]
        mismatches = [
            record["question"]
            for ref, record in zip(baseline, records)
            if ref["answer"] != record["answer"]
        ]
        result["comparisons"][mode] = {
            "answer_mismatches": mismatches,
            "exact_text_matches": sum(
                ref["text"] == record["text"] for ref, record in zip(baseline, records)
            ),
            "length_capped": sum(
                record["finish_reason"] == "length" for record in records
            ),
        }
    if "fixed" in arms and "adaptive" in arms:
        pairs = list(zip(arms["fixed"]["records"], arms["adaptive"]["records"]))
        result["fixed_vs_adaptive"] = {
            "answer_mismatches": [
                ref["question"]
                for ref, record in pairs
                if ref["answer"] != record["answer"]
            ],
            "exact_text_matches": sum(
                ref["text"] == record["text"] for ref, record in pairs
            ),
        }
    return result


if __name__ == "__main__":
    main()
