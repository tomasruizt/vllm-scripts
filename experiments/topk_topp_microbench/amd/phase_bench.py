"""Cost of single phases of the New Triton filter, each in its own launch.

- pass: one streaming pass over a row (_mass_upto), by 1 program and by one per row.
- alone: the single-program cut of a row from its histogram (_resolve_alone).
Usage: python phase_bench.py [--rows 8] [--bins 8192 256] [--warps 16]
"""

import argparse

import torch
import triton
import triton.language as tl
from vllm.v1.sample.ops import topk_topp_triton_hier as hier


@triton.jit
def _pass_kernel(logits_ptr, out_ptr, vocab, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    total = hier._mass_upto(logits_ptr + row.to(tl.int64) * vocab, 0.0, 0x7FFFFFFFFFFFFFFF, vocab, BLOCK)
    tl.store(out_ptr + row, total)


@triton.jit
def _alone_kernel(logits_ptr, hist_ptr, scratch_ptr, state_ptr, cut_ptr, stats_ptr, k, p, vocab,
                  BINS: tl.constexpr, BIN_SCALE: tl.constexpr, WINDOW: tl.constexpr,
                  BLOCK: tl.constexpr):  # fmt: skip
    row = tl.program_id(0)
    hier._resolve_alone(logits_ptr + row.to(tl.int64) * vocab, hist_ptr + row * BINS,
                        scratch_ptr + row.to(tl.int64) * WINDOW, state_ptr + row, cut_ptr + row,
                        k, p, tl.load(stats_ptr + 2 * row).to(tl.float32), tl.load(stats_ptr + 2 * row + 1),
                        vocab, BINS, BIN_SCALE, WINDOW, BLOCK)  # fmt: skip


@triton.jit
def _gather_kernel(logits_ptr, scratch_ptr, found_ptr, more_ptr, window_ptr, span_ptr, vocab, chunk, splits,
                   BINS: tl.constexpr, WINDOW: tl.constexpr, BLOCK: tl.constexpr):  # fmt: skip
    row = tl.program_id(0) // splits
    hier._gather_part(logits_ptr + row.to(tl.int64) * vocab, scratch_ptr + row.to(tl.int64) * WINDOW,
                      found_ptr + row, more_ptr + 2 * row, 0.0, window_ptr, span_ptr,
                      tl.program_id(0) % splits * chunk, vocab, chunk, BINS, WINDOW, BLOCK)  # fmt: skip


def bench_gather(logits: torch.Tensor, splits: int, warps: int, lo: int) -> None:
    """The gather pass alone: bins lo to lo of a first span (row max 0), no k."""
    rows, vocab = logits.shape
    bins = hier.BINS
    scratch = torch.empty(rows * hier.WINDOW, dtype=torch.int64, device="cuda")
    found = torch.zeros(rows, dtype=torch.int32, device="cuda")
    more = torch.zeros(rows * 2, dtype=torch.float64, device="cuda")
    window = torch.tensor([lo, lo, bins, 4096], dtype=torch.int32, device="cuda")
    span = torch.tensor([0.0, -float("inf"), hier.BIN_SCALE, 0.0], device="cuda")
    chunk = triton.cdiv(triton.cdiv(vocab, splits), 4096) * 4096
    grid = (rows * splits,)
    args = (logits.clone(), scratch, found, more, window, span, vocab, chunk, splits, bins, hier.WINDOW, 4096)
    _gather_kernel[grid](*args, num_warps=warps)
    hits = found.tolist()
    ms = triton.testing.do_bench(lambda: (found.zero_(), _gather_kernel[grid](*args, num_warps=warps)))
    print(f"PHASE gather rows={rows} splits={splits} warps={warps} bin={lo} hits/row={hits[0]} us={1000 * ms:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--bins", type=int, nargs="+", default=[8192, 256])
    parser.add_argument("--warps", type=int, nargs="+", default=[16])
    parser.add_argument("--logits-file", default="qwen36-verify-logits-t1.pt")
    parser.add_argument("--gather", action="store_true")
    args = parser.parse_args()
    if args.gather:
        ranks = torch.argsort(torch.rand(8, 248320, device="cuda"), dim=1).float() + 1
        zipf = -1.1 * ranks.log() + 1.1 * 1.0  # row max 0... at rank 1
        zipf = -1.1 * ranks.log()
        for warps in args.warps:
            for lo in [40, 85, 127]:
                bench_gather(zipf, 32, warps, lo * hier.BINS // 128)
        return
    real = torch.load(args.logits_file).float().cuda()[: args.rows].contiguous()
    vocab = real.shape[1]
    ranks = torch.argsort(torch.rand(args.rows, vocab, device="cuda"), dim=1).float() + 1
    for name, logits in {"real": real, "zipf": -1.1 * ranks.log()}.items():
        for warps in args.warps:
            bench_pass(name, logits, warps)
            for bins in args.bins:
                for k, p in [(vocab, 0.95), (10000, 0.9)]:
                    bench_alone(name, logits, warps, bins, k, p)


def bench_pass(name: str, logits: torch.Tensor, warps: int) -> None:
    rows, vocab = logits.shape
    out = torch.empty(rows, dtype=torch.float64, device="cuda")
    for grid in [(1,), (rows,)]:
        ms = triton.testing.do_bench(lambda: _pass_kernel[grid](logits, out, vocab, 4096, num_warps=warps))
        print(f"PHASE {name} pass warps={warps} programs={grid[0]} us={1000 * ms:.1f}")


def bench_alone(name: str, logits: torch.Tensor, warps: int, bins: int, k: int, p: float) -> None:
    rows, vocab = logits.shape
    row_max = logits.max(dim=1, keepdim=True).values
    stats = torch.stack([row_max[:, 0].double(), (logits - row_max).exp().double().sum(1)], 1)
    scale = bins / 16.0
    b = ((row_max - logits) * scale).clamp(max=bins - 1).long()
    hist = torch.stack([torch.bincount(r, minlength=bins) for r in b]).int().flatten()
    scratch = torch.empty(rows * hier.WINDOW, dtype=torch.int64, device="cuda")
    state = torch.empty(rows, dtype=torch.int32, device="cuda")
    cut = torch.empty(rows, dtype=torch.int64, device="cuda")
    args = (logits, hist, scratch, state, cut, stats.flatten().contiguous(), k, p, vocab, bins, scale,
            hier.WINDOW, 4096)  # fmt: skip
    kernel = _alone_kernel[(rows,)](*args, num_warps=warps)
    ms = triton.testing.do_bench(lambda: _alone_kernel[(rows,)](*args, num_warps=warps))
    states = state.tolist()
    print(f"PHASE {name} alone warps={warps} bins={bins} k={k} p={p} us={1000 * ms:.1f} "
          f"states={sorted(set(states))} n_regs={kernel.n_regs} spills={kernel.n_spills}")  # fmt: skip


if __name__ == "__main__":
    main()
