"""Phase stamps (s_memrealtime, 100 MHz) of the new kernel's persistent filter on real
rows, top-p 0.95. Usage: python phase_trace.py LOGITS.pt ROWS... (in the ROCm image)"""
import os
import sys
from collections import Counter

import torch
import importlib

from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton as main_triton

m = importlib.import_module(next((a[9:] for a in sys.argv if a.startswith("--module=")), "trace_pr_kernel"))
SHUFFLE = "--shuffle" in sys.argv
FLUSH = torch.empty(1 << 30, dtype=torch.int8, device="cuda")
NAMES = {-1: "noop", -2: "hist", -4: "self", -5: "gather", -6: "alone"}


def main() -> None:
    real = torch.load(sys.argv[1]).float().cuda()
    for rows in map(int, [a for a in sys.argv[2:] if not a.startswith("--")]):
        run(real, rows)


def run(real: torch.Tensor, rows: int) -> None:
    logits = real.repeat(-(-rows // real.shape[0]), 1)[:rows].contiguous()
    if SHUFFLE:
        logits = logits[torch.randperm(rows, generator=torch.Generator().manual_seed(0)).cuda()].contiguous()
    p = torch.full((rows,), 0.95, device="cuda")
    if "--partial" in sys.argv:  # like topp_partial: the second half unfiltered
        p[rows // 2 :] = 1.0
    for _ in range(3):
        m.apply_top_k_top_p_triton(logits.clone(), None, p)
    plan = m._PLANS[(logits.device, *logits.shape, logits.stride(0), False, True)]
    full = timed(lambda x: m.apply_top_k_top_p_triton(x, None, p), logits)
    levels = timed(lambda x: launch(plan[:1], x, p), logits)
    main_us = timed(lambda x: main_triton(x, None, p), logits)
    b = m._BUFFERS[(logits.device, logits.shape[1])]
    b["trace"].zero_()
    x = logits.clone()
    m.apply_top_k_top_p_triton(x, None, p)
    torch.cuda.synchronize()
    t = b["trace"].tolist()
    states = Counter(NAMES.get(s, "tier" if s >= 0 else s) for s in b["state"][:rows].tolist())
    kernel, grid = plan[1][0], plan[1][1]
    print(f"ROWS {rows} programs={t[34]} grid={grid} levels_grid={plan[0][1]} rounds={t[33]} "
          f"hsplits={t[32]} n_regs={kernel.n_regs} shared={kernel.metadata.shared} "
          f"capacity={dict(m._CAPACITY)} states={dict(states)} kept_mean={(x > -1e30).sum(1).float().mean():.0f}")
    print(f"  main_triton={main_us:.1f}us full={full:.1f}us levels={levels:.1f}us filter_span={(t[35] - t[0]) / 100:.1f}us")
    us = lambda a, b_: round((t[b_] - t[a]) / 100, 1) if t[a] and t[b_] else None
    print(f"  select={us(0, 1)}")
    prev = 1
    for r in range(t[33]):
        s = 2 + 6 * r
        print(f"  round{r}: hist_rows={t[s + 4]} cut_rows={t[s + 5]} hist={us(prev, s)} plan={us(s, s + 1)} "
              f"gather={us(s + 1, s + 2)} cut={us(s + 2, s + 3)}")
        prev = s + 3
    print(f"  fill={(t[35] - t[prev]) / 100:.1f}", flush=True)


def launch(entries, x, p):
    stream = torch.cuda.current_stream().cuda_stream
    for kernel, grid, args, (at_logits, at_k, at_p) in entries:
        args[at_logits], args[at_p] = x, p
        kernel.run(*grid, stream, kernel.function, kernel.packed_metadata, None, None, None, *args)


def timed(fn, logits, iters: int = 20) -> float:
    """Mean µs per call, clone excluded, cold caches."""
    total = 0.0
    for _ in range(iters):
        x = logits.clone()
        FLUSH.zero_()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn(x)
        end.record()
        torch.cuda.synchronize()
        total += start.elapsed_time(end)
    return 1000 * total / iters


if __name__ == "__main__":
    main()
