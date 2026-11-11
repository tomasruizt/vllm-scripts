"""Triton core-path prototype (triton_core.py) vs. Triton, CUDA and the optimized
FlashInfer fast path (fi_opt.py, only on the scenarios it supports), all checked
against the exact reference.

Usage, from the PR tree: python triton_core_bench.py OUT.jsonl --logits-file ROWS.pt
"""

import argparse
from pathlib import Path

import torch
from fastpath_bench import BATCHES, supported
from fi_opt import register
from large_k import load_benchmark
from triton_core import apply_top_k_top_p_triton_core


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--logits-file", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=BATCHES)
    parser.add_argument("--implementations", nargs="+", default=["triton-core", "cuda", "triton"])
    parser.add_argument("--fast-path", default="fi-opt", help="added on supported scenarios")
    args = parser.parse_args()
    bench = load_benchmark(Path("benchmarks/benchmark_topk_topp.py"))
    real_rows = torch.load(args.logits_file).float().cuda()
    configs = bench.create_benchmark_configs(args.batch_sizes, [131072, 248320])
    configs += bench.create_real_configs(args.batch_sizes, real_rows.shape[1])
    register(bench)
    register_core(bench)
    for config in configs:
        names = args.implementations + ([args.fast_path] if args.fast_path and supported(config) else [])
        bench.run_benchmark([config], names, check=True, output=args.output, real_rows=real_rows)


def register_core(bench) -> None:
    """Pass the core path the same host-side hint as the Triton kernel it falls back to."""
    implementation = bench.implementation

    def with_core(name: str, config):
        if name != "triton-core":
            return implementation(name, config)
        hint = bench.has_topp_only_rows(config)
        return lambda logits, k, p: apply_top_k_top_p_triton_core(logits, k, p, hint)

    bench.IMPLEMENTATIONS["triton-core"] = apply_top_k_top_p_triton_core
    bench.implementation = with_core


if __name__ == "__main__":
    main()
