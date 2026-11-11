"""Run parameter coverage through the existing benchmark's actual CLI entry point."""

import argparse
from contextlib import redirect_stdout
import hashlib
from importlib.metadata import version
import itertools
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--preset", choices=["expanded", "reduced", "eligible"], default="expanded")
    parser.add_argument("--implementations", nargs="+", default=["triton", "flashinfer-fastpath", "pytorch"],
                        choices=["triton", "flashinfer-fastpath", "pytorch"])
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--skip", type=int, nargs="*", default=[])
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()
    plans = list(make_plans(args.preset))
    selected = select_plans(plans, args.shard, args.shards, args.skip)
    selected = [(i, p) for i, p in selected if i >= args.start_index]
    records = sum(len(p["batches"]) * len(p["vocabs"]) * len(p["scenarios"]) for p in plans)
    summary = dict(invocations=len(plans), records=records, timed_cells=records * len(args.implementations),
                   selected_invocations=len(selected), shard=args.shard, shards=args.shards)
    summary["selected_records"] = sum(len(p["batches"]) * len(p["vocabs"]) * len(p["scenarios"]) for _, p in selected)
    print(json.dumps(summary), flush=True)
    if args.plan_only:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plan.json").write_text(json.dumps(plans, indent=2))
    (args.output / "metadata.json").write_text(json.dumps(metadata(), indent=2))
    warnings.filterwarnings("error", message=".*Falling back to CUDA events.*")
    started = time.monotonic()
    for index, plan in selected:
        execute(args.output, plan, index, args.iterations, args.implementations)
    summary["elapsed_s"] = time.monotonic() - started
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "DONE").write_text("ok\n")


def select_plans(plans: list[dict], shard: int, shards: int, skip: list[int]) -> list:
    groups = {}
    for index, item in enumerate(plans):
        key = json.dumps({k: v for k, v in item.items() if k != "repeat"}, sort_keys=True)
        groups.setdefault(key, []).append(index)
    loads = [0] * shards
    assignments = {}
    weight = lambda ids: sum(len(plans[i]["batches"]) * len(plans[i]["vocabs"]) * len(plans[i]["scenarios"]) for i in ids)
    for ids in sorted(groups.values(), key=weight, reverse=True):
        target = min(range(shards), key=lambda i: loads[i])
        for index in ids:
            assignments[index] = target
        loads[target] += weight(ids)
    return [(i, p) for i, p in enumerate(plans) if assignments[i] == shard and i not in skip]


def make_plans(preset: str = "expanded"):
    all_scenarios = ["topk_whole", "topk_partial", "topp_whole", "topp_partial", "topk_topp_whole", "mixed_partial"]
    if preset == "reduced":
        for repeat in range(2):
            yield plan("reduced", repeat, [4, 64, 512], [32768, 131072, 248320], None, None, scenarios=all_scenarios)
        return
    if preset == "eligible":
        for repeat in range(3):
            yield plan("eligible", repeat, [4, 64, 512], [131072, 248320], None, None,
                       scenarios=["topk_whole", "topk_topp_whole"])
        return
    batches = [1, 4, 9, 32, 128, 280, 512]
    vocabs = [4096, 32768, 131072, 151936, 248320, 262144]
    for repeat in range(2):
        yield plan("coverage", repeat, batches, vocabs, 20, 0.8, scenarios=all_scenarios)
        for k, p in itertools.product([1, 2, 20, 32, 64, 100, 128, 129, 256], [0.01, 0.1, 0.5, 0.8, 0.95, 1.0]):
            scenarios = ["topk_topp_whole", "topk_whole"] if p == 0.01 else ["topk_topp_whole"]
            yield plan("parameters", repeat, [1, 4, 128, 512], [32768, 248320], k, p, scenarios=scenarios)
    distributions = [("zipf", 13), ("zipf", 128), ("normal", None), ("bf16-rounded", None)]
    for (distribution, finite), k, p in itertools.product(distributions, [1, 20, 128, 256], [0.1, 0.8, 0.95, 1.0]):
        scenarios = ["topk_topp_whole", "topk_whole"] if p == 0.1 else ["topk_topp_whole"]
        yield plan("inputs", 0, [1, 4, 128, 512], [4096, 131072, 248320, 262144], k, p, distribution, finite, scenarios)
    for k, p in itertools.product([1, 20, 128, 129], [0.1, 0.8, 1.0]):
        scenarios = ["topk_topp_whole", "topk_whole"] if p == 0.1 else ["topk_topp_whole"]
        yield plan("odd", 0, [1, 4, 32], [257, 4097, 32769], k, p, scenarios=scenarios)


def plan(phase: str, repeat: int, batches: list[int], vocabs: list[int], k: int | None, p: float | None,
         distribution: str = "zipf", finite: int | None = None, scenarios: list[str] | None = None) -> dict:
    return dict(phase=phase, repeat=repeat, batches=batches, vocabs=vocabs, top_k=k, top_p=p,
                distribution=distribution, finite=finite, scenarios=scenarios)


def execute(output: Path, item: dict, index: int, iterations: int, implementations: list[str]) -> None:
    prefix = output / f"case-{index:03d}"
    order = list(implementations)
    if item["repeat"] % 2:
        order.reverse()
    command = ["benchmark_topk_topp.py", "--batch-sizes", *map(str, item["batches"]),
               "--vocab-sizes", *map(str, item["vocabs"]), "--distribution", item["distribution"],
               "--warmup-iters", "5", "--benchmark-iters", str(iterations),
               "--implementations", *order, "--scenarios", *item["scenarios"],
               "--check", "--output", str(prefix.with_suffix(".jsonl"))]
    if item["top_k"] is not None:
        command += ["--top-k", str(item["top_k"])]
    if item["top_p"] is not None:
        command += ["--top-p", str(item["top_p"])]
    if item["finite"] is not None:
        command += ["--num-finite-logits", str(item["finite"])]
    status = dict(index=index, plan=item, command=command, state="running")
    (output / "status.json").write_text(json.dumps(status, indent=2))
    prefix.with_suffix(".command.json").write_text(json.dumps(command, indent=2))
    print(json.dumps(dict(index=index, **item)), flush=True)
    original = sys.argv
    sys.argv = command
    try:
        with prefix.with_suffix(".log").open("x") as log, redirect_stdout(log):
            runpy.run_path("benchmark_topk_topp.py", run_name="__main__")
    finally:
        sys.argv = original
    status["state"] = "complete"
    (output / "status.json").write_text(json.dumps(status, indent=2))


def metadata() -> dict:
    import torch
    from vllm.v1.sample.ops import topk_topp_triton, topk_topp_sampler

    modules = [Path("benchmark_topk_topp.py"), Path(topk_topp_triton.__file__), Path(topk_topp_sampler.__file__)]
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in modules}
    source = subprocess.run(["git", "-C", str(modules[1].parent), "rev-parse", "HEAD"], text=True, capture_output=True)
    commit = source.stdout.strip() if source.returncode == 0 else f"wheel:{version('vllm')}"
    packages = {name: version(name) for name in ["torch", "triton", "flashinfer-python", "cupti-python", "pandas"]}
    return dict(gpu=torch.cuda.get_device_name(), gpu_ids=os.environ.get("CUDA_VISIBLE_DEVICES"),
                python=sys.executable, cuda=torch.version.cuda, packages=packages, commit=commit, sources=sources)


if __name__ == "__main__":
    main()
