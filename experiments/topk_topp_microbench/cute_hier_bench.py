"""CuTe hierarchical kernel (rh-4spj) vs. CUDA hierarchical (hier_cuda.py), the CuTe port and the
CUDA filter (#66521), on benchmark_topk_topp.py's synthetic and real-logit configs.

Usage, from the CuTe tree: python cute_hier_bench.py OUT.jsonl --logits-file ROWS.pt
    [--batch-sizes ...] [--vocab-sizes ...] [--scenarios real_topp ...] [--implementations ...]
"""

import argparse
from pathlib import Path

import torch
from large_k import large_k_configs, load_benchmark

from hier_cuda import apply_top_k_top_p_hier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--logits-file", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[131072, 248320])
    parser.add_argument("--scenarios", nargs="+", help="name prefixes, e.g. real_topp topk_whole")
    parser.add_argument("--large-k", action="store_true", help="only large_k.py's configs")
    parser.add_argument("--implementations", nargs="+", default=["cute-hier", "hier-cuda", "cute", "cuda"])
    args = parser.parse_args()
    bench = load_benchmark(Path("benchmarks/benchmark_topk_topp.py"))
    bench.IMPLEMENTATIONS["hier-cuda"] = apply_top_k_top_p_hier
    real_rows = torch.load(args.logits_file).float().cuda()
    configs = bench.create_benchmark_configs(args.batch_sizes, args.vocab_sizes)
    configs += bench.create_real_configs(args.batch_sizes, real_rows.shape[1])
    if args.large_k:
        configs = large_k_configs(bench, [1, 8, 64, 256, 1024])
    if args.scenarios:
        configs = [c for c in configs if any(c.name.startswith(s + "_b") for s in args.scenarios)]
    bench.run_benchmark(configs, args.implementations, check=True, output=args.output, real_rows=real_rows)


if __name__ == "__main__":
    main()
