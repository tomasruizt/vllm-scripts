"""Run eligible sampling profiles through the existing vLLM benchmark CLI."""

import argparse
from contextlib import redirect_stdout
from functools import partial
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import subprocess
import time
import warnings

from sweep_existing import metadata


PROFILES = [
    dict(name="uniform_topk20", k=[20], p=None),
    dict(name="uniform_topk100", k=[100], p=None),
    dict(name="heterogeneous_topk", k=[20, 100], p=None),
    dict(name="uniform_combined20", k=[20], p=[0.8]),
    dict(name="uniform_combined100", k=[100], p=[0.95]),
    dict(name="heterogeneous_combined", k=[20, 100], p=[0.8, 0.95]),
    dict(name="mixed_topk_combined", k=[20, 100], p=[1.0, 0.8]),
]
BATCHES = [1, 4, 16, 64, 128, 256, 512]
VOCABS = [65536, 131072, 151936, 248320]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    implementations = ["triton", "flashinfer-fastpath"]
    profiles = PROFILES
    if args.ablation:
        implementations = ["triton", "fi-baseline", "fi-no-flip", "fi-unsorted", "flashinfer-fastpath"]
        profiles = [p for p in PROFILES if p["name"] in {"uniform_topk20", "heterogeneous_topk", "heterogeneous_combined"}]
        BATCHES[:] = [4, 64, 512]
        VOCABS[:] = [65536, 131072, 248320]
    plans = [dict(repeat=r, profile=p, batches=batches(p))
             for r, p in itertools.product(range(3), profiles)]
    cells = sum(len(p["batches"]) * len(VOCABS) for p in plans)
    print(json.dumps(dict(unique_cases=cells // 3, passes=3, timing_cells=cells * len(implementations))), flush=True)
    if args.plan_only:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plan.json").write_text(json.dumps(plans, indent=2))
    spec = importlib.util.spec_from_file_location("scenario_benchmark", "benchmark_topk_topp.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if args.ablation:
        install_ablation(module)
    info = metadata()
    info["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.ablation:
        info["sources"].update({p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                for p in ["baseline.py", "no_flip.py", "unsorted.py"]})
    (args.output / "metadata.json").write_text(json.dumps(info, indent=2))
    warnings.filterwarnings("error", message=".*Falling back to CUDA events.*")
    if args.validate:
        subprocess.run([sys.executable, "validate_existing.py", "--output", str(args.output / "oracle"),
                        "--benchmark", "benchmark_topk_topp.py"], check=True)
        subprocess.run([sys.executable, "validate_adapter.py", "--output", str(args.output / "edges")], check=True)
    started = time.monotonic()
    for index, plan in enumerate(plans):
        execute(module, args.output, plan, index, implementations)
    (args.output / "summary.json").write_text(json.dumps(dict(
        unique_cases=cells // 3, timing_cells=cells * len(implementations), elapsed_s=time.monotonic() - started)))
    (args.output / "DONE").write_text("ok\n")


def batches(profile: dict) -> list[int]:
    return [b for b in BATCHES if b > 1 or len(profile["k"]) == 1]


def execute(module, output: Path, plan: dict, index: int, implementations: list[str]) -> None:
    module.create_benchmark_configs = partial(configs, module, plan["profile"])
    order = implementations.copy()
    if len(order) > 2:
        rotate = (index + plan["repeat"]) % len(order)
        order = order[rotate:] + order[:rotate]
    if (plan["repeat"] + index % len(PROFILES)) % 2:
        order.reverse()
    prefix = output / f"case-{index:03d}"
    command = ["benchmark_topk_topp.py", "--batch-sizes", *map(str, plan["batches"]),
               "--vocab-sizes", *map(str, VOCABS), "--implementations", *order,
               "--warmup-iters", "5", "--benchmark-iters", "30", "--check",
               "--output", str(prefix.with_suffix(".jsonl"))]
    prefix.with_suffix(".command.json").write_text(json.dumps(command, indent=2))
    status = dict(index=index, plan=plan, order=order)
    (output / "status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status), flush=True)
    sys.argv = command
    with prefix.with_suffix(".log").open("x") as log, redirect_stdout(log):
        module.main()


def install_ablation(module) -> None:
    import runpy

    original_prepare = module.prepare_kwargs
    module.prepare_kwargs = lambda name, k, p, vocab: original_prepare(
        "flashinfer-fastpath" if name.startswith("fi-") else name, k, p, vocab)
    for name, filename in [("fi-baseline", "baseline.py"), ("fi-no-flip", "no_flip.py"),
                           ("fi-unsorted", "unsorted.py")]:
        snapshot = runpy.run_path(filename)
        module.IMPLEMENTATIONS[name] = snapshot["apply_top_k_top_p_flashinfer_fastpath"]


def configs(module, profile: dict, batch_sizes: list[int], vocab_sizes: list[int], **_) -> list:
    import torch

    def rows(values: list, batch: int, dtype):
        return torch.tensor([values[i % len(values)] for i in range(batch)], device="cuda", dtype=dtype)

    result = []
    for vocab, batch in itertools.product(vocab_sizes, batch_sizes):
        k = rows(profile["k"], batch, torch.int32)
        p = None if profile["p"] is None else rows(profile["p"], batch, torch.float32)
        ops = module.calculate_ops_pct(k, p, vocab, batch)
        result.append(module.BenchmarkConfig(
            name=f'{profile["name"]}_b{batch}_v{vocab}', batch_size=batch,
            vocab_size=vocab, k_values=k, p_values=p, description=profile["name"], ops_pct=ops))
    return result


if __name__ == "__main__":
    main()
