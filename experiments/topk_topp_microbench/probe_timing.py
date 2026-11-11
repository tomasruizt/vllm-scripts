"""Compare timing methods on a few real_topk_topp cases (why fixed timing moves small batches).

Usage (tree with main's Triton, code dir on PYTHONPATH): python probe_timing.py FIXED_BENCH.py QWEN.pt
"""

import sys
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti
from large_k import load_benchmark

FLUSH = 1 << 30


def main() -> None:
    bench = load_benchmark(Path(sys.argv[1]))
    bench.IMPLEMENTATIONS["final"] = __import__("final_kernel").apply_top_k_top_p_triton
    rows = torch.load(sys.argv[2]).float().cuda()
    for batch in (8, 64, 2048):
        config = [c for c in bench.create_real_configs([batch], rows.shape[1]) if c.name.startswith("real_topk_topp")][0]
        logits = bench.tile_rows(rows, batch)
        for name in ("triton", "final"):
            func = bench.implementation(name, config)
            print(batch, name, " ".join(f"{m}={t:.1f}" for m, t in methods(bench, func, logits, config).items()), flush=True)


def methods(bench, func, logits, config) -> dict[str, float]:
    k, p = config.k_values, config.p_values
    mean = lambda ts: 1000 * sum(ts) / len(ts)  # noqa: E731
    clone = mean(cupti(lambda: logits.clone()))
    return {
        "cupti_clone_minus": mean(cupti(lambda: func(logits.clone(), k, p))) - clone,
        "fixed": mean(bench.bench_gpu_time(func, logits, k, p, 5, 20)),
        "fixed_sync": mean(events(func, logits, k, p, sync=True)),
        "fixed_256m": mean(events(func, logits, k, p, flush_bytes=256 << 20)),
        "fixed_sleep": mean(events(func, logits, k, p, sleep=True)),
        "fixed_256m_sleep": mean(events(func, logits, k, p, flush_bytes=256 << 20, sleep=True)),
    }


def cupti(fn) -> list[float]:
    return bench_gpu_time_with_cupti(fn, dry_run_iters=5, repeat_iters=20, use_cuda_graph=False, cold_l2_cache=True)


def events(func, logits, k, p, sync=False, flush_bytes=FLUSH, sleep=False) -> list[float]:
    """sleep: a GPU spin after the flush, so the CPU is ahead and launch gaps stay hidden."""
    fresh, flush = torch.empty_like(logits), torch.empty(flush_bytes, dtype=torch.int8, device="cuda")
    times = []
    for _ in range(25):
        fresh.copy_(logits)
        flush.zero_()
        torch.cuda.synchronize() if sync else None
        torch.cuda._sleep(2_000_000) if sleep else None
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        func(fresh, k, p)
        end.record()
        times.append((start, end))
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in times[5:]]


if __name__ == "__main__":
    main()
