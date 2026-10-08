"""Run paired, stratified SPEED-Bench subsets sized to roughly 30 seconds."""

import argparse
import json
import math
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import median

from benchmark import profile, reset_prefix_cache, run_mode


def subset(records, count, *, seed):
    groups = defaultdict(list)
    for record in records:
        groups[record["category"]].append(record)
    rng = random.Random(seed)
    selected = []
    for index, category in enumerate(sorted(groups)):
        n = count // len(groups) + (index < count % len(groups))
        selected.extend(rng.sample(groups[category], n))
    rng.shuffle(selected)
    return selected


def write_subset(path, records):
    content = "".join(json.dumps(record) + "\n" for record in records)
    if path.exists():
        assert path.read_text() == content, path
    else:
        path.write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-repo", type=Path, required=True)
    parser.add_argument("--reclaim-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--target-seconds", type=float, default=30)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=8044)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.concurrencies = [8, 16, 32, 64, 128, 256]
    args.resume = True
    assert args.target_seconds > 0 and args.num_runs >= 2
    for repo, expected in [
        (args.baseline_repo, "58b329845"),
        (args.reclaim_repo, "b39cfba61"),
    ]:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        assert sha.startswith(expected), (repo, sha)
        assert not subprocess.check_output(["git", "diff", "HEAD"], cwd=repo)
    records = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    assert len(records) == 1536
    assert all(len(r["messages"]) == 1 for r in records)
    root = args.output.resolve()
    args.output = root
    datasets = root / "workloads"
    datasets.mkdir(parents=True, exist_ok=True)
    plan_path = root / "workload-plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        assert (
            plan["seed"] == args.seed and plan["target_seconds"] == args.target_seconds
        )
        assert plan["num_runs"] == args.num_runs
    else:
        reference = json.loads((args.reference / "result.json").read_text())
        assert reference["status"] == "completed"
        counts = {}
        for c in args.concurrencies:
            durations = [
                json.loads((args.reference / r["aiperf_summary"]).read_text())[
                    "benchmark_duration"
                ]["avg"]
                for r in reference["runs"]
                if r["concurrency"] == c
            ]
            counts[str(c)] = min(
                len(records),
                max(
                    c, math.ceil(len(records) * args.target_seconds / median(durations))
                ),
            )
        plan = {
            "seed": args.seed,
            "target_seconds": args.target_seconds,
            "num_runs": args.num_runs,
            "counts": counts,
            "calibration": {},
            "method_order": (
                f"avon repeats 1-{args.num_runs // 2}, "
                "reclaim all repeats, avon remaining repeats"
            ),
        }

    def save_plan():
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")

    save_plan()
    print("Initial requests per concurrency:", plan["counts"], flush=True)
    print(
        f"Measured inference budget: "
        f"{2 * len(args.concurrencies) * args.num_runs * args.target_seconds / 60:.0f}"
        " minutes, plus calibration/startup/client overhead",
        flush=True,
    )

    def workload(repeat, concurrency):
        count = plan["counts"][str(concurrency)]
        path = datasets / f"c{concurrency}-r{repeat}.jsonl"
        write_subset(
            path, subset(records, count, seed=args.seed + 1000 * repeat + concurrency)
        )
        return path, count

    def calibrate(run_args, env):
        original = run_args.dataset
        try:
            for c in args.concurrencies:
                if str(c) in plan["calibration"]:
                    continue
                count = plan["counts"][str(c)]
                label = f"calibration-c{c}"
                path = datasets / f"{label}.jsonl"
                write_subset(path, subset(records, count, seed=args.seed + c))
                run_args.dataset = path
                reset = reset_prefix_cache(run_args)
                (run_args.output / f"{label}-cache-reset.json").write_text(
                    json.dumps(reset) + "\n"
                )
                print(f"Calibration c={c}: {count} requests", flush=True)
                report = json.loads(profile(run_args, label, c, count, env).read_text())
                duration = report["benchmark_duration"]["avg"]
                adjusted = min(
                    len(records), max(c, round(count * args.target_seconds / duration))
                )
                plan["counts"][str(c)] = adjusted
                plan["calibration"][str(c)] = {
                    "requests": count,
                    "seconds": duration,
                    "adjusted_requests": adjusted,
                }
                save_plan()
                print(
                    f"Calibration c={c}: {duration:.1f}s; "
                    f"using {adjusted} requests per measurement",
                    flush=True,
                )
        finally:
            run_args.dataset = original

    if args.prepare_only:
        for c in args.concurrencies:
            workload(1, c)
        return
    split = args.num_runs // 2
    blocks = [
        ("avon", args.baseline_repo, range(1, split + 1)),
        ("reclaim", args.reclaim_repo, range(1, args.num_runs + 1)),
        ("avon", args.baseline_repo, range(split + 1, args.num_runs + 1)),
    ]
    for method, repo, repeats in blocks:
        result_path = root / method / "result.json"
        completed = set()
        if result_path.exists():
            result = json.loads(result_path.read_text())
            completed = {(r["repeat"], r["concurrency"]) for r in result["runs"]}
        if all((r, c) in completed for r in repeats for c in args.concurrencies):
            continue
        run_mode(
            args,
            method,
            repo,
            records,
            repeats=repeats,
            workload=workload,
            prepare=calibrate,
        )
    print("Both methods completed all measurements", flush=True)


if __name__ == "__main__":
    main()
