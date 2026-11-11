"""The report's 140 cases (synthetic Zipf at V131k/V248k plus real Qwen3.6 rows) on
ROCm, checked against the exact reference. "clone-only" times the input copy, which
the report subtracts. "triton-hier" is the New Triton draft (topk_topp_triton_hier.py).

Usage, in the ROCm image (amd_run.sh):
  python amd_bench.py OUT.jsonl --logits-file ROWS.pt --implementations triton clone-only
"""

import argparse
from pathlib import Path

import torch
from large_k import load_benchmark

BATCHES = [1, 4, 8, 16, 64, 128, 256, 512, 1024, 2048]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--logits-file", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=BATCHES)
    parser.add_argument("--implementations", nargs="+", default=["triton", "clone-only"])
    parser.add_argument("--benchmark-iters", type=int, default=20)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bench = load_benchmark(Path("benchmark_topk_topp.py"))
    register(bench, args.implementations)
    real_rows = torch.load(args.logits_file).float().cuda()
    configs = bench.create_benchmark_configs(args.batch_sizes, [131072, 248320])
    configs += bench.create_real_configs(args.batch_sizes, real_rows.shape[1])
    bench.run_benchmark(configs, args.implementations, benchmark_iters=args.benchmark_iters,
                        check=True, output=args.output, real_rows=real_rows)  # fmt: skip


def register(bench, names: list[str]) -> None:
    bench.IMPLEMENTATIONS["clone-only"] = lambda logits, k, p: logits
    if "triton-hier" in names:
        from vllm.v1.sample.ops.topk_topp_triton_hier import apply_top_k_top_p_triton_hier

        bench.IMPLEMENTATIONS["triton-hier"] = apply_top_k_top_p_triton_hier
    check = bench.check_implementations

    def check_filters(logits, config, implementations):
        """clone-only does not filter, so it has nothing to check."""
        return check(logits, config, [n for n in implementations if n != "clone-only"])

    bench.check_implementations = check_filters


if __name__ == "__main__":
    main()
