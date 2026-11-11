"""Cost of a per-row 2048-bin histogram of (row max - logit) in 1/64 nat bins, with
tl.histogram per tile and one atomic add per bin per program.

Usage: python triton_core_hist.py
"""

import torch
from torch.profiler import ProfilerActivity, profile

from vllm.triton_utils import tl, triton


@triton.jit
def _hist_kernel(x_ptr, hist_ptr, vocab, chunk, row_max, BINS: tl.constexpr, BLOCK: tl.constexpr,
                 MODE: tl.constexpr):  # fmt: skip
    row = tl.program_id(0)
    lo = tl.program_id(1) * chunk
    hi = tl.minimum(lo + chunk, vocab)
    offs = tl.arange(0, BLOCK)
    total = tl.zeros([BINS], tl.int32)
    for start in range(lo, hi, BLOCK):
        idx = start + offs
        x = tl.load(x_ptr + row.to(tl.int64) * vocab + idx, mask=idx < hi, other=-float("inf"))
        b = tl.minimum((row_max - x) * 1024.0, BINS - 1.0).to(tl.int32)
        if MODE == 0:
            total += tl.histogram(b, BINS, mask=idx < hi)
        else:
            tl.atomic_add(hist_ptr + row * BINS + b, 1, mask=idx < hi, sem="relaxed")
    if MODE == 0:
        tl.atomic_add(hist_ptr + row * BINS + tl.arange(0, BINS), total, sem="relaxed")


def main() -> None:
    vocab = 248320
    for rows, splits in ((1, 32), (16, 32), (2048, 1)):
        x = torch.randn(rows, vocab, device="cuda") * 2
        hist = torch.zeros(rows * 16384, dtype=torch.int32, device="cuda")
        chunk = triton.cdiv(triton.cdiv(vocab, splits), 4096) * 4096
        for bins, block, warps in ((2048, 4096, 8), (8192, 4096, 8), (16384, 4096, 8), (16384, 2048, 8)):
            run = lambda: _hist_kernel[(rows, splits)](x, hist, vocab, chunk, 10.0, BINS=bins, BLOCK=block, MODE=0, num_warps=warps)
            print(rows, splits, bins, block, warps, time(run), flush=True)


def time(run) -> float:
    run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            run()
        torch.cuda.synchronize()
    events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return round(sum(e.device_time for e in events) / 10, 1)


if __name__ == "__main__":
    main()
