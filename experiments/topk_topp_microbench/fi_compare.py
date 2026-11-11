"""Triton vs CUDA vs FlashInfer top-k/top-p masking on benchmark_topk_topp.py configs.

FlashInfer variants:
  flashinfer-mask: main's benchmark helper (top_k_mask_logits, softmax, top_p_renorm_probs).
  flashinfer-fastpath: vLLM PR #65649 (FlashInfer top-k selection, compact top-p), only
    for uniform-range k in 1..256 and vocab >= 65536. Its host top-k bounds are read
    outside timing.
Run from a vLLM tree with the CUDA filter: python fi_compare.py --output out.jsonl
"""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

sys.path.insert(0, "benchmarks")
import benchmark_topk_topp as bench  # noqa: E402
from fi_fastpath import (  # noqa: E402
    apply_top_k_top_p_flashinfer,
    can_apply_top_k_top_p_flashinfer,
)

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch  # noqa: E402

IMPLS = ["triton", "cuda", "flashinfer-mask", "flashinfer-fastpath"]


def main() -> None:
    args = parse_args()
    configs = bench.create_benchmark_configs(args.batch_sizes, args.vocab_sizes)
    with args.output.open("w") as out:
        for config in configs:
            for row in measure(config):
                out.write(json.dumps(row) + "\n")
                print(row, flush=True)


def measure(config) -> list[dict]:
    logits = bench.create_logits(config.batch_size, config.vocab_size)
    k, p = config.k_values, config.p_values
    reference = apply_top_k_top_p_pytorch(logits.clone(), k, p).softmax(dim=-1)
    rows = []
    for name in IMPLS:
        func = implementation(name, logits, k)
        row = dict(scenario=config.name.split("_b")[0], batch=config.batch_size,
                   vocab=config.vocab_size, impl=name, us=None, max_tv=None)
        if func is not None:
            row["max_tv"] = max_tv(func(logits.clone(), k, p), reference)
            row["us"] = 1000 * time_ms(func, logits, k, p)
        rows.append(row)
    return rows


def implementation(name: str, logits: torch.Tensor, k: torch.Tensor | None) -> Callable | None:
    if name != "flashinfer-fastpath":
        return bench.IMPLEMENTATIONS[name]
    bounds = (min(k.tolist()), max(k.tolist())) if k is not None else None
    if bounds is not None and bounds[0] <= 0:
        bounds = None
    if not can_apply_top_k_top_p_flashinfer(logits, k, bounds):
        return None
    return lambda x, k, p: apply_top_k_top_p_flashinfer(x, k, p, bounds)


def max_tv(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return (actual.softmax(dim=-1) - reference).abs().sum(-1).max().item() / 2


def time_ms(func: Callable, logits: torch.Tensor, k, p) -> float:
    times = bench_gpu_time_with_cupti(
        lambda: func(logits.clone(), k, p),
        dry_run_iters=5, repeat_iters=20, use_cuda_graph=False, cold_l2_cache=True,
    )
    return sum(times) / len(times)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=[1, 4, 16, 64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[32768, 131072, 248320])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
