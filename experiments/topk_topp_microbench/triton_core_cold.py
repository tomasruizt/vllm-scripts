"""Kernel times, launch gaps and filter phases of New Triton under the gate's conditions:
L2 flushed and the GPU idle before each call, the call including logits.clone().

Usage, from the PR tree: python triton_core_cold.py --logits-file ROWS.pt [--only topp_whole_b1_ ...]
"""

import argparse
import statistics
from pathlib import Path

import torch
import triton_core
from gate import gate_configs
from large_k import load_benchmark
from torch.profiler import ProfilerActivity, profile
from triton_core_phases import PHASES

FLUSH = torch.empty(512 * 2**20, dtype=torch.uint8, device="cuda")  # > L2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-file", required=True)
    parser.add_argument("--only", nargs="*", default=["topp_whole_b1_", "topp_whole_b8_", "real_topp_b8_"])
    args = parser.parse_args()
    bench = load_benchmark(Path("benchmarks/benchmark_topk_topp.py"))
    real_rows = torch.load(args.logits_file).float().cuda()
    for config in gate_configs(bench, real_rows):
        if any(s in config.name for s in args.only):
            logits = bench.tile_rows(real_rows, config.batch_size) if config.real_logits else \
                bench.create_logits(config.batch_size, config.vocab_size)  # fmt: skip
            run = lambda: triton_core.apply_top_k_top_p_triton_core(logits.clone(), config.k_values, config.p_values)  # noqa: E731
            print(config.name, "cold:", kernels(run), flush=True)
            print(config.name, "warm:", kernels(run, flush=False), flush=True)
            print(config.name, "cold phases:", phases(run), flush=True)
            print(config.name, "warm phases:", phases(run, flush=False), flush=True)


def kernels(run, n: int = 20, flush: bool = True) -> str:
    """Median duration of each kernel and of the gaps between them, and the span (µs)."""
    run()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            if flush:
                FLUSH.zero_()
            torch.cuda.synchronize()
            run()
            torch.cuda.synchronize()
    events = [e for e in prof.events() if e.device_type.name == "CUDA" and "zero" not in e.name.lower()
              and "fill" not in e.name.lower()]  # fmt: skip
    per_call = len(events) // n
    rows = [events[i * per_call:(i + 1) * per_call] for i in range(n)]
    parts = []
    for j in range(per_call):
        name = rows[0][j].name.split("_kernel")[0][-12:]
        parts.append(f"{name} {statistics.median(r[j].time_range.elapsed_us() for r in rows):.1f}")
        if j + 1 < per_call:
            gap = statistics.median(r[j + 1].time_range.start - r[j].time_range.end for r in rows)
            parts.append(f"gap {gap:.1f}")
    span = statistics.median(r[-1].time_range.end - r[0].time_range.start for r in rows)
    return ", ".join(parts) + f", span {span:.1f}"


def phases(run, n: int = 10, flush: bool = True) -> dict:
    """Median end of each filter phase (µs from the filter kernel's start)."""
    ends = []
    for _ in range(n):
        triton_core.TIMINGS = torch.zeros(16 * 4096, dtype=torch.int64, device="cuda")
        if flush:
            FLUSH.zero_()
        torch.cuda.synchronize()
        run()
        torch.cuda.synchronize()
        t = triton_core.TIMINGS.view(-1, 16)
        t = t[t[:, 0] > 0].double() / 1000
        ends.append({p: (t[:, i].max() - t[:, 0].min()).item() for p, i in PHASES if t[:, i].max() > 0})
    triton_core.TIMINGS = None
    return {p: round(statistics.median(e[p] for e in ends), 1) for p in ends[0]}


if __name__ == "__main__":
    main()
