"""Time FlashInfer's separate renorm kernels against Triton and the PR fast path."""

import argparse
from contextlib import redirect_stdout
from functools import partial
import importlib.util
import itertools
import json
from pathlib import Path
import sys
from types import ModuleType

from renorm_impls import RENORM_IMPLEMENTATIONS
from sweep_existing import metadata

DISABLED_K = "vocab"
PROFILES = [
    dict(name="uniform_topk20", k=[20], p=None),
    dict(name="uniform_topk100", k=[100], p=None),
    dict(name="heterogeneous_topk", k=[20, 100], p=None),
    dict(name="uniform_combined20", k=[20], p=[0.8]),
    dict(name="uniform_combined100", k=[100], p=[0.95]),
    dict(name="heterogeneous_combined", k=[20, 100], p=[0.8, 0.95]),
    dict(name="mixed_topk_combined", k=[20, 100], p=[1.0, 0.8]),
    dict(name="uniform_topp", k=None, p=[0.8]),
    dict(name="topp_partial", k=None, p=[0.8, 1.0]),
    dict(name="topk_disabled_rows", k=[20, DISABLED_K], p=[0.8, 0.8]),
]
BATCHES = [1, 4, 16, 64, 256, 512]
VOCABS = [32768, 131072, 248320]
PASSES = 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    module = load_benchmark()
    (args.output / "metadata.json").write_text(json.dumps(metadata(), indent=2))
    for index, plan in enumerate(make_plans(args.smoke)):
        execute(module, args.output, plan, index)
    (args.output / "DONE").write_text("ok\n")


def make_plans(smoke: bool) -> list[dict]:
    if smoke:
        cases = [(0, PROFILES[3], 131072), (0, PROFILES[8], 131072)]
        return [make_plan(*case, batches=[4]) for case in cases]
    return [make_plan(*case) for case in itertools.product(range(PASSES), PROFILES, VOCABS)]


def make_plan(repeat: int, profile: dict, vocab: int, batches: list[int] | None = None) -> dict:
    uniform = all(len(values) == 1 for values in (profile["k"] or [0], profile["p"] or [0]))
    batches = batches or [b for b in BATCHES if b > 1 or uniform]
    return dict(repeat=repeat, profile=profile, vocab=vocab, batches=batches)


def load_benchmark() -> ModuleType:
    spec = importlib.util.spec_from_file_location("renorm_benchmark", "benchmark_topk_topp.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.IMPLEMENTATIONS.update(RENORM_IMPLEMENTATIONS)
    return module


def implementations(plan: dict, index: int) -> list[str]:
    """Rotate and reverse order across passes. The fast path only runs on eligible batches."""
    k = plan["profile"]["k"]
    eligible = k is not None and DISABLED_K not in k and plan["vocab"] >= 65536
    names = ["triton", *RENORM_IMPLEMENTATIONS, *(["flashinfer-fastpath"] if eligible else [])]
    rotate = (index + plan["repeat"]) % len(names)
    names = names[rotate:] + names[:rotate]
    return names[::-1] if plan["repeat"] % 2 else names


def execute(module: ModuleType, output: Path, plan: dict, index: int) -> None:
    module.create_benchmark_configs = partial(make_configs, module, plan["profile"])
    command = ["benchmark_topk_topp.py", "--batch-sizes", *map(str, plan["batches"]),
               "--vocab-sizes", str(plan["vocab"]), "--implementations", *implementations(plan, index),
               "--warmup-iters", "5", "--benchmark-iters", "30", "--check",
               "--output", str(output / f"case-{index:03d}.jsonl")]
    (output / f"case-{index:03d}.plan.json").write_text(json.dumps(dict(plan=plan, command=command)))
    print(json.dumps(dict(index=index, repeat=plan["repeat"], name=plan["profile"]["name"], vocab=plan["vocab"])), flush=True)
    sys.argv = command
    with (output / f"case-{index:03d}.log").open("x") as log, redirect_stdout(log):
        module.main()


def make_configs(module: ModuleType, profile: dict, batch_sizes: list[int], vocab_sizes: list[int], **_) -> list:
    configs = []
    for vocab, batch in itertools.product(vocab_sizes, batch_sizes):
        k, p = row_tensors(profile, batch, vocab)
        configs.append(module.BenchmarkConfig(
            name=f'{profile["name"]}_b{batch}_v{vocab}', batch_size=batch, vocab_size=vocab, k_values=k,
            p_values=p, description=profile["name"], ops_pct=module.calculate_ops_pct(k, p, vocab, batch)))
    return configs


def row_tensors(profile: dict, batch: int, vocab: int) -> tuple:
    import torch

    def rows(values: list | None, dtype):
        if values is None:
            return None
        resolved = [vocab if value == DISABLED_K else value for value in values]
        return torch.tensor([resolved[i % len(resolved)] for i in range(batch)], device="cuda", dtype=dtype)

    return rows(profile["k"], torch.int32), rows(profile["p"], torch.float32)


if __name__ == "__main__":
    main()
