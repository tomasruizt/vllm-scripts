"""Host overhead of New Triton at small batch: host µs per call (no sync), GPU span of one
call (CUDA events) and the sum of its kernels' times (profiler), on real rows.

Usage, from the PR tree: python triton_core_host.py --logits-file ROWS.pt
"""

import argparse
import time

import torch
import triton_core
from torch.profiler import ProfilerActivity, profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-file", required=True)
    args = parser.parse_args()
    rows = torch.load(args.logits_file).float().cuda()
    for batch in (1, 8, 64):
        logits = rows.repeat(-(-batch // rows.shape[0]), 1)[:batch].contiguous()
        p = torch.full((batch,), 0.95, device="cuda")
        run = lambda: triton_core.apply_top_k_top_p_triton_core(logits, None, p)  # noqa: E731
        jit = lambda: (triton_core._PLANS.clear(), run())  # noqa: E731  (Triton's launcher every call)
        print(f"b{batch}: host {host_us(run):.1f} µs/call (JIT launcher {host_us(jit):.1f}), span {span_us(run):.1f} µs "
              f"(JIT launcher {span_us(jit):.1f}), kernels {kernel_us(run):.1f} µs", flush=True)


def host_us(run, n: int = 200) -> float:
    run()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n):
        run()
    elapsed = time.perf_counter() - start
    torch.cuda.synchronize()
    return 1e6 * elapsed / n


def span_us(run, n: int = 20) -> float:
    """Median GPU time from before to after one call, the GPU idle before it."""
    times = []
    for _ in range(n):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        times.append(1000 * start.elapsed_time(end))
    return sorted(times)[n // 2]


def kernel_us(run, n: int = 20) -> float:
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            run()
        torch.cuda.synchronize()
    return sum(e.device_time_total for e in prof.key_averages()) / n


if __name__ == "__main__":
    main()
