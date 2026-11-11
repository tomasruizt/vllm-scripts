"""Per-kernel device time of the masking ops on benchmark_topk_topp.py configs.

Usage: kernel_times.py [--impls cuda,triton] scenario:batch:vocab ...
"""

import argparse
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, "benchmarks")
from benchmark_topk_topp import create_benchmark_configs, create_logits  # noqa: E402

from vllm.v1.sample.ops.topk_topp_cuda import apply_top_k_top_p_cuda  # noqa: E402
from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton  # noqa: E402

IMPLS = {"cuda": apply_top_k_top_p_cuda, "triton": apply_top_k_top_p_triton}
ITERS = 50


def config(scenario: str, batch: int, vocab: int):
    configs = create_benchmark_configs([batch], [vocab])
    return next(c for c in configs if c.name.startswith(f"{scenario}_b"))


def total_us(func, logits, k, p, cold: bool, warm_code: bool) -> float:
    """Device time per call. cold flushes L2 before each call; warm_code then
    reruns the op on a copy of the first row, so only its data stays cold."""
    copies = [logits.clone() for _ in range(ITERS + 5)]
    flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    tiny = [logits[:1].clone() for _ in range(ITERS)]
    tk, tp = (None if t is None else t[:1] for t in (k, p))
    for c in copies[:5]:
        func(c, k, p)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for c, t in zip(copies[5:], tiny):
            if cold:
                flush.zero_()
            if warm_code:
                func(t, tk, tp)
            func(c, k, p)
        torch.cuda.synchronize()
    events = [e for e in prof.key_averages() if e.device_time_total > 0]
    events = [e for e in events if "topk" in e.key or "topp" in e.key]
    us = sum(e.device_time_total for e in events) / ITERS
    return us - (total_us(func, logits[:1], tk, tp, False, False) if warm_code else 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impls", default=",".join(IMPLS), help="comma-separated")
    parser.add_argument("--cold", action="store_true", help="flush L2 per call")
    parser.add_argument("--warm-code", action="store_true")
    parser.add_argument("cases", nargs="+", help="scenario:batch:vocab")
    args = parser.parse_args()
    for case in args.cases:
        scenario, batch, vocab = case.split(":")
        c = config(scenario, int(batch), int(vocab))
        logits = create_logits(c.batch_size, c.vocab_size)
        for name in args.impls.split(","):
            us = total_us(
                IMPLS[name], logits, c.k_values, c.p_values, args.cold, args.warm_code
            )
            print(f"{case} {name} {us:8.1f}us")


if __name__ == "__main__":
    main()
