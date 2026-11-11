"""Cost of the histogram pass ((row max - logit) bins, added to global bins) per
method, BINS and warps: rows x splits programs, each histograms its chunk.
Methods: tl.histogram (as the kernel), or one global atomic per logit.

Usage: python hist_bench.py [--rows 256] [--vocab 248320] [--splits 4]
"""

import argparse
import itertools

import torch
import triton
import triton.language as tl
from vllm.v1.sample.ops import topk_topp_triton_hier as hier


@triton.jit
def _hist_kernel(logits_ptr, hist_ptr, vocab, chunk, splits, BINS: tl.constexpr,
                 BIN_SCALE: tl.constexpr, BLOCK: tl.constexpr, ATOMIC: tl.constexpr):  # fmt: skip
    row = tl.program_id(0) // splits
    lo = tl.program_id(0) % splits * chunk
    row_ptr = logits_ptr + row.to(tl.int64) * vocab
    if not ATOMIC:
        offs = tl.arange(0, BLOCK)
        hi = tl.minimum(lo + chunk, vocab)
        counts = tl.zeros([BINS], tl.int32)
        x_next = tl.load(row_ptr + lo + offs, mask=lo + offs < hi, other=-float("inf"))
        for start in range(lo, hi, BLOCK):
            idx = start + offs
            x = x_next
            x_next = tl.load(row_ptr + idx + BLOCK, mask=idx + BLOCK < hi, other=-float("inf"))
            counts += tl.histogram(hier._bin(x, 0.0, BIN_SCALE, BINS), BINS, mask=idx < hi)
        tl.atomic_add(hist_ptr + row * BINS + tl.arange(0, BINS), counts, mask=counts > 0, sem="relaxed", scope="gpu")
    else:
        offs = tl.arange(0, BLOCK)
        hi = tl.minimum(lo + chunk, vocab)
        rest = 0
        for start in range(lo, hi, BLOCK):
            idx = start + offs
            x = tl.load(row_ptr + idx, mask=idx < hi, other=-float("inf"))
            b = hier._bin(x, 0.0, BIN_SCALE, BINS)
            tl.atomic_add(hist_ptr + row * BINS + b, 1, mask=(idx < hi) & (b < BINS - 1),
                          sem="relaxed", scope="gpu")  # fmt: skip
            rest += tl.sum(((idx < hi) & (b == BINS - 1)).to(tl.int32))
        tl.atomic_add(hist_ptr + row * BINS + BINS - 1, rest, sem="relaxed", scope="gpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--vocab", type=int, default=248320)
    parser.add_argument("--splits", type=int, default=4)
    parser.add_argument("--bins", type=int, nargs="+", default=[1024, 8192])
    parser.add_argument("--warps", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--atomic", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--block", type=int, nargs="+", default=[4096])
    args = parser.parse_args()
    shape = (args.rows, args.vocab)
    ranks = torch.argsort(torch.rand(shape, device="cuda"), dim=1).float() + 1
    data = {"uniform": -torch.rand(shape, device="cuda") * 16, "zipf": -1.1 * ranks.log()}
    for name, logits in data.items():
        for atomic, bins, warps, block in itertools.product(args.atomic, args.bins, args.warps, args.block):
            run(name, logits, args.splits, bool(atomic), bins, warps, block)


def run(name, logits, splits: int, atomic: bool, bins: int, warps: int, block: int) -> None:
    rows, vocab = logits.shape
    hist = torch.zeros(rows * bins, dtype=torch.int32, device="cuda")
    chunk = triton.cdiv(triton.cdiv(vocab, splits), block) * block
    grid = (rows * splits,)
    args = (logits, hist, vocab, chunk, splits, bins, bins / 16.0, block, atomic)
    kernel = _hist_kernel[grid](*args, num_warps=warps)
    ms = triton.testing.do_bench(lambda: _hist_kernel[grid](*args, num_warps=warps))
    print(f"HIST {name} atomic={int(atomic)} bins={bins} warps={warps} block={block} "
          f"us={1000 * ms:.1f} n_regs={kernel.n_regs} spills={kernel.n_spills}")  # fmt: skip


if __name__ == "__main__":
    main()
