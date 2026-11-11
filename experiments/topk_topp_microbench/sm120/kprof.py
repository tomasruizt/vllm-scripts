"""Per-kernel device times (torch profiler) of kernel copies on gate cases, with an L2 flush before each call.

Usage, from the PR tree: python kprof.py --logits-file ROWS.pt --only topp_partial_b256 --modules base_triton topk_topp_triton
"""

import argparse
import importlib
from collections import defaultdict
from pathlib import Path

import torch
from gate import gate_configs
from large_k import load_benchmark
from torch.profiler import ProfilerActivity, profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-file", required=True)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--modules", nargs="*", default=["base_triton", "topk_topp_triton"])
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    bench = load_benchmark(Path("benchmarks/benchmark_topk_topp.py"))
    real_rows = torch.load(args.logits_file).float().cuda()
    flush = torch.empty(256 << 20, dtype=torch.int8, device="cuda")
    for c in gate_configs(bench, real_rows):
        if not any(s in c.name for s in args.only):
            continue
        logits = bench.tile_rows(real_rows, c.batch_size) if c.real_logits else bench.create_logits(c.batch_size, c.vocab_size)
        for name in args.modules:
            fn = importlib.import_module(name).apply_top_k_top_p_triton
            print(c.name, name, kernel_times(fn, logits, c.k_values, c.p_values, flush, args.iters), flush=True)


def kernel_times(fn, logits, k, p, flush, iters: int) -> str:
    for _ in range(3):
        fn(logits.clone(), k, p)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            flush.zero_()
            fn(logits.clone(), k, p)
        torch.cuda.synchronize()
    total = defaultdict(float)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            total[e.name[:40]] += e.device_time / iters
    return ", ".join(f"{n} {t:.1f}" for n, t in sorted(total.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    main()
