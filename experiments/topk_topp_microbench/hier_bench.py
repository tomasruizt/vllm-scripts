"""Hierarchical CUDA filter (hier_cuda.py) vs. the CUDA filter, Triton and the Triton
core path, all checked against the exact reference. The optimized FlashInfer fast path
(fi_opt.py) runs on the scenarios it supports.

Usage, from the PR tree: python hier_bench.py OUT.jsonl --logits-file ROWS.pt
    [--batch-sizes ...] [--scenarios real_topp ...] [--implementations ...]
"""

import argparse
from pathlib import Path

import torch
from fastpath_bench import BATCHES, supported
from fi_opt import register
from large_k import load_benchmark
from triton_core_bench import register_core

from hier_cuda import apply_top_k_top_p_hier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--logits-file", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=BATCHES)
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[131072, 248320])
    parser.add_argument("--scenarios", nargs="+", help="name prefixes, e.g. real_topp topk_whole")
    parser.add_argument("--implementations", nargs="+", default=["hier", "cuda", "triton", "triton-core"])
    parser.add_argument("--fast-path", default="fi-opt", help="added on supported scenarios")
    args = parser.parse_args()
    bench = load_benchmark(Path("benchmarks/benchmark_topk_topp.py"))
    real_rows = torch.load(args.logits_file).float().cuda()
    configs = bench.create_benchmark_configs(args.batch_sizes, args.vocab_sizes)
    configs += bench.create_real_configs(args.batch_sizes, real_rows.shape[1])
    if args.scenarios:
        configs = [c for c in configs if any(c.name.startswith(s + "_b") for s in args.scenarios)]
    register(bench)
    register_core(bench)
    bench.IMPLEMENTATIONS.update(HIER)
    for config in configs:
        names = args.implementations + ([args.fast_path] if args.fast_path and supported(config) else [])
        bench.run_benchmark([config], names, check=True, output=args.output, real_rows=real_rows)


# The time breakdown: streaming alone, and streaming plus the -inf fill.
HIER = {
    "hier": apply_top_k_top_p_hier,
    "hier-stream": lambda logits, k, p: apply_top_k_top_p_hier(logits, k, p, ablate=1),
    "hier-stream-fill": lambda logits, k, p: apply_top_k_top_p_hier(logits, k, p, ablate=2),
}


if __name__ == "__main__":
    main()
