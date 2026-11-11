"""Time Triton vs. the CUDA filter (and its CuTe DSL port) for top-k in the low thousands (V248k).

Usage, from the PR tree: python large_k.py OUT.jsonl [--batch-sizes ...]
"""

import argparse
import importlib.util
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 64, 256, 1024])
    parser.add_argument("--implementations", nargs="+", default=["triton", "cuda"])
    args = parser.parse_args()
    bench = load_benchmark(Path("benchmarks/benchmark_topk_topp.py"))
    configs = large_k_configs(bench, args.batch_sizes)
    bench.run_benchmark(configs, args.implementations, check=True, output=args.output)


def load_benchmark(path: Path):
    spec = importlib.util.spec_from_file_location("benchmark_topk_topp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def large_k_configs(bench, batch_sizes: list[int]) -> list:
    """top-k only and top-k with p=0.9 for k in the low thousands, V248k."""
    return [config(bench, b, k, p) for b in batch_sizes for k in (1000, 4096, 10000) for p in (None, 0.9)]


def config(bench, batch: int, k: int, p: float | None):
    return bench.BenchmarkConfig(
        name=f"k{k}{'_p0.9' if p else ''}_b{batch}_v248k",
        batch_size=batch,
        vocab_size=248320,
        k_values=torch.full((batch,), k, dtype=torch.int32, device="cuda"),
        p_values=None if p is None else torch.full((batch,), p, device="cuda"),
        description=f"k={k} p={p}",
    )


if __name__ == "__main__":
    main()
