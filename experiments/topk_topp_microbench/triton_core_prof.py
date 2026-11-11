"""Per-kernel GPU times of the Triton core prototype, on real rows (top-k 20, top-p 0.95).

Usage, from the PR tree: python triton_core_prof.py --logits-file ROWS.pt [--batch-sizes ...]
"""

import argparse
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile
from triton_core import apply_top_k_top_p_triton_core


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-file", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 256, 2048])
    parser.add_argument("--no-k", action="store_true")
    args = parser.parse_args()
    rows = torch.load(args.logits_file).float().cuda()
    for batch in args.batch_sizes:
        print(batch, "k" if not args.no_k else "p-only", profile_batch(rows, batch, not args.no_k), flush=True)


def profile_batch(rows: torch.Tensor, batch: int, with_k: bool) -> dict[str, float]:
    logits = rows.repeat(-(-batch // rows.shape[0]), 1)[:batch].contiguous()
    k = torch.full((batch,), 20, dtype=torch.int32, device="cuda") if with_k else None
    p = torch.full((batch,), 0.95, device="cuda")
    copies = [logits.clone() for _ in range(10)]
    apply_top_k_top_p_triton_core(logits.clone(), k, p, not with_k)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for copy in copies:
            apply_top_k_top_p_triton_core(copy, k, p, not with_k)
        torch.cuda.synchronize()
    times = defaultdict(float)
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            times[event.name[:30]] += event.device_time / len(copies)
    return {name: round(t, 1) for name, t in times.items()}


if __name__ == "__main__":
    main()
