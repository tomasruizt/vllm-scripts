"""Exactness of the hierarchical CUDA filter (hier_cuda.py) against the FP64 reference,
over Zipf, real, tied (bf16-rounded, constant) and odd-vocab rows, all split counts.

Usage, from the PR tree: python hier_test.py [--logits-file ROWS.pt] [--quick]
"""

import argparse
import itertools
from pathlib import Path

import torch
from large_k import load_benchmark

import hier_cuda
from hier_cuda import apply_top_k_top_p_hier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-file")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    bench = load_benchmark(Path("benchmarks/benchmark_topk_topp.py"))
    torch.manual_seed(0)
    rows = {name: make(bench, name) for name in ROWS}
    if args.logits_file:
        rows["real"] = torch.load(args.logits_file).float().cuda()[:16]
    params = list(itertools.product(KS, PS))
    splits = [0, 1, 4] if args.quick else [0, 1, 2, 4, 8, 16]
    failures = sum(run_case(bench, n, x, k, p, s) for n, x in rows.items() for k, p in params for s in splits)
    print(f"FAILURES {failures}")


ROWS = ["zipf248k", "zipf131k", "bf16_zipf", "normal50001", "constant", "small1000", "neginf_tail"]
KS = [None, 1, 20, 50, 1000, 10000, "mixed"]
PS = [None, 0.5, 0.95, 0.99, "mixed"]


def make(bench, name: str) -> torch.Tensor:
    if name == "zipf248k":
        return bench.create_logits(4, 248320)
    if name == "zipf131k":
        return bench.create_logits(3, 131072)
    if name == "bf16_zipf":
        return (bench.create_logits(3, 248320) * 4).bfloat16().float()
    if name == "normal50001":
        return torch.randn(5, 50001, device="cuda") * 3
    if name == "constant":
        return torch.zeros(2, 65536, device="cuda")
    if name == "small1000":
        return torch.randn(3, 1000, device="cuda")
    x = torch.randn(3, 248320, device="cuda") * 2
    x[:, 1000:] = -float("inf")
    return x


def run_case(bench, name, x, k, p, splits) -> int:
    rows, vocab = x.shape
    kt = None if k is None else row_values(k, rows, vocab, torch.int32)
    pt = None if p is None else row_values(p, rows, vocab, torch.float32)
    if kt is None and pt is None:
        return 0
    hier_cuda.SPLITS = splits
    out = apply_top_k_top_p_hier(x.clone(), kt, pt)
    torch.cuda.synchronize()
    ref = bench.exact_reference(x, kt, pt)
    kept, ref_kept = out > -float("inf"), ref > -float("inf")
    bad = (kept != ref_kept).sum(-1)
    close = torch.allclose(out.softmax(-1), ref.softmax(-1), rtol=1e-5, atol=1e-6)
    if bad.sum() == 0 and torch.equal(out[kept], ref[ref_kept]):
        return 0
    print(f"MISMATCH {name} k={k} p={p} splits={splits}: rows {bad.tolist()} "
          f"kept {kept.sum(-1).tolist()} ref {ref_kept.sum(-1).tolist()} close={close}")  # fmt: skip
    return 1


def row_values(v, rows: int, vocab: int, dtype) -> torch.Tensor:
    if v == "mixed":
        choices = [vocab, 5, 50, 200] if dtype == torch.int32 else [1.0, 0.3, 0.9, 0.999]
        v = [choices[i % len(choices)] for i in range(rows)]
    else:
        v = [v] * rows
    return torch.tensor(v, dtype=dtype, device="cuda")


if __name__ == "__main__":
    main()
