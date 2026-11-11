"""New Triton correctness and path check on edge cases vs. the exact reference.

Usage, from the PR tree: python triton_core_check.py --logits-file ROWS.pt
"""

import argparse
from pathlib import Path

import torch
import triton_core
from large_k import load_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-file", required=True)
    args = parser.parse_args()
    bench = load_benchmark(Path("benchmarks/benchmark_topk_topp.py"))
    real = torch.load(args.logits_file).float().cuda()
    zipf = bench.create_logits(8, 131072)
    huge = torch.randn(4, 131072, device="cuda")
    huge[:, 0] = 1e9
    tied = torch.randn(4, 131072, device="cuda").to(torch.bfloat16).float()
    cases = [
        ("real p0.95", real, None, 0.95), ("real k20 p0.95", real, 20, 0.95),
        ("real k200", real, 200, None), ("real k1000 p0.9", real, 1000, 0.9),
        ("real k5000", real, 5000, None), ("zipf p0.9", zipf, None, 0.9),
        ("zipf k100 p0.9", zipf, 100, 0.9), ("huge k1000", huge, 1000, None),
        ("huge p0.5", huge, None, 0.5), ("tied k300", tied, 300, None),
        ("tied p0.99", tied, None, 0.99), ("small real b3 p0.95", real[:3], None, 0.95),
    ]  # fmt: skip
    for name, logits, k, p in cases:
        print(name, check(bench, logits, k, p), flush=True)
    for name, kernel in triton_core.LAUNCHED.items():
        if isinstance(kernel, int):
            continue
        print(name, "regs", kernel.n_regs, "spills", kernel.n_spills, "smem", kernel.metadata.shared)


def check(bench, logits: torch.Tensor, k: int | None, p: float | None) -> dict:
    rows = logits.shape[0]
    kt = None if k is None else torch.full((rows,), k, dtype=torch.int32, device="cuda")
    pt = None if p is None else torch.full((rows,), p, device="cuda")
    reference = bench.exact_reference(logits, kt, pt).softmax(-1)
    actual = triton_core.apply_top_k_top_p_triton_core(logits.clone(), kt, pt).softmax(-1)
    state = triton_core._BUFFERS[next(reversed(triton_core._BUFFERS))]["state"]
    paths = {int(v): int(c) for v, c in zip(*state.unique(return_counts=True))}
    tv = (actual - reference).abs().sum(-1).max().item() / 2
    exact = torch.allclose(actual, reference, rtol=1e-5, atol=1e-6)
    return dict(exact=exact, tv=f"{tv:.2e}", paths=paths)


if __name__ == "__main__":
    main()
