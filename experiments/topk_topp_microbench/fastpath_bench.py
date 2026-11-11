"""Time the FlashInfer fast path (fi_fastpath.py, from #65649) on the benchmark
scenarios it supports: every row with the same active k <= 256 and vocab >= 64Ki.

Usage, from the PR tree: python fastpath_bench.py OUT.jsonl --logits-file ROWS.pt
"""

import argparse
from pathlib import Path

import torch
from fi_fastpath import apply_top_k_top_p_flashinfer, can_apply_top_k_top_p_flashinfer
from large_k import load_benchmark

BATCHES = [1, 4, 8, 16, 64, 128, 256, 512, 1024, 2048]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--logits-file", required=True)
    args = parser.parse_args()
    bench = load_benchmark(Path("benchmarks/benchmark_topk_topp.py"))
    real_rows = torch.load(args.logits_file).float().cuda()
    configs = bench.create_benchmark_configs(BATCHES, [131072, 248320])
    configs += bench.create_real_configs(BATCHES, real_rows.shape[1])
    configs = [c for c in configs if supported(c)]
    register(bench)
    bench.run_benchmark(configs, ["fi-fastpath"], check=True, output=args.output, real_rows=real_rows)


def supported(config) -> bool:
    k = config.k_values
    if k is None:
        return False
    bounds = (int(k.min()), int(k.max()))
    logits = torch.empty(config.batch_size, config.vocab_size, device="meta")
    return can_apply_top_k_top_p_flashinfer(logits, k, bounds)


def register(bench) -> None:
    """The fast path needs the host-side k bounds, computed outside the timing."""
    implementation = bench.implementation

    def with_fastpath(name: str, config):
        if name != "fi-fastpath":
            return implementation(name, config)
        bounds = (int(config.k_values.min()), int(config.k_values.max()))
        return lambda logits, k, p: apply_top_k_top_p_flashinfer(logits, k, p, bounds)

    bench.IMPLEMENTATIONS["fi-fastpath"] = apply_top_k_top_p_flashinfer
    bench.implementation = with_fastpath


if __name__ == "__main__":
    main()
