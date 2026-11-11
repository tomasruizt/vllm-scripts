"""Per-row path of the histogram rounds on one batch: state, span, above, window.

Usage: python debug_rounds.py [--rows 8] [--k 0] [--p 0.9] [--real]
"""

import argparse
import os
import time

os.environ["HIER_TRACE"] = "1"

import torch
from vllm.v1.sample.ops import topk_topp_triton_hier as hier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=248320)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--cases", nargs="+", help="several runs: ROWS:K:P[:real] each")
    args = parser.parse_args()
    for case in args.cases or []:
        rows, k, p, *real = case.split(":")
        args.rows, args.k, args.p, args.real = int(rows), int(k), float(p), bool(real)
        print(f"CASE {case}")
        run(args)
    if not args.cases:
        run(args)


def run(args) -> None:
    shape = (args.rows, args.vocab)
    if args.real:
        rows = torch.load("qwen36-verify-logits-t1.pt").float().cuda()
        logits = rows.repeat(-(-args.rows // rows.shape[0]), 1)[: args.rows].contiguous()
    else:
        ranks = torch.argsort(torch.rand(shape, device="cuda"), dim=1).float() + 1
        logits = -1.1 * ranks.log()
    k = torch.full((args.rows,), args.k, dtype=torch.int32, device="cuda") if args.k else None
    p = torch.full((args.rows,), args.p, device="cuda") if args.p < 1 else None
    for _ in range(3):
        x = logits.clone()
        torch.cuda.synchronize()
        start = time.perf_counter()
        hier.apply_top_k_top_p_triton_hier(x, k, p)
        torch.cuda.synchronize()
        print(f"call ms={1000 * (time.perf_counter() - start):.2f}")
    wrong = (reference(logits, args.k, args.p) != (x > -float("inf"))).sum(1).tolist()
    print(f"EXACT mismatches per row: {wrong}")
    b = hier._BUFFERS[(logits.device, args.vocab)]
    t = b["trace"].tolist()
    tick = 0.01 if hier.current_platform.is_rocm() else 0.001  # µs per tick
    marks = {i: round((t[i] - t[0]) * tick, 1) for i in range(41) if t[i] >= t[0] > 0}
    print(f"TRACE µs since start: {marks}")
    for row in range(min(args.rows, 2) if "span" in b else 0):
        span = b["span"][4 * row : 4 * row + 4].tolist()
        print(f"row {row} state={b['state'][row].item()} span={[round(v, 4) for v in span]} "
              f"above={b['above'][2 * row : 2 * row + 2].tolist()} window={b['window'][4 * row : 4 * row + 4].tolist()} "
              f"found={b['found'][row].item()} kept={(x[row] > -float('inf')).sum().item()}")  # fmt: skip


def reference(logits: torch.Tensor, k: int, p: float) -> torch.Tensor:
    """Kept mask: the top k (ties by lowest index), then the smallest prefix of mass p."""
    values, order = logits.double().sort(dim=1, descending=True, stable=True)
    keep = torch.ones_like(values, dtype=torch.bool)
    if k:
        keep[:, k:] = False
    probs = torch.where(keep, (values - values[:, :1]).exp(), 0.0)
    if p < 1:
        before = probs.cumsum(1) - probs
        keep &= before < p * probs.sum(1, keepdim=True)
    return torch.zeros_like(keep).scatter(1, order, keep)


if __name__ == "__main__":
    main()
