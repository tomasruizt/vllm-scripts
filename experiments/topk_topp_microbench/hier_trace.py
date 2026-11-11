"""Phase clocks of the first CTA and per-row selection paths of the hierarchical CUDA
filter (hier_cuda.py), on the benchmark scenarios.

Usage, from the PR tree: python hier_trace.py --logits-file ROWS.pt [--batch-sizes ...]
"""

import argparse
import collections
from pathlib import Path

import torch
from large_k import load_benchmark

import hier_cuda
from hier_cuda import apply_top_k_top_p_hier, splits

PHASES = ["stream", "stats", "level2", "level1", "gather", "finish", "barrier", "fill", "barrier2", "scatter"]  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-file", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 64, 256, 2048])
    parser.add_argument("--scenarios", nargs="+", default=["real_topk_topp", "real_topp", "topk_whole", "topp_whole"])  # fmt: skip
    parser.add_argument("--splits", type=int, nargs="+", default=[0])
    parser.add_argument("--rows", type=int, nargs="+", default=[0], help="rows to trace")
    args = parser.parse_args()
    bench = load_benchmark(Path("benchmarks/benchmark_topk_topp.py"))
    real_rows = torch.load(args.logits_file).float().cuda()
    configs = bench.create_benchmark_configs(args.batch_sizes, [248320])
    configs += bench.create_real_configs(args.batch_sizes, real_rows.shape[1])
    configs = [c for c in configs if any(c.name.startswith(s + "_b") for s in args.scenarios)]
    mhz = getattr(torch.cuda.get_device_properties(0), "clock_rate", 1_980_000) / 1000  # kHz
    for config in configs:
        logits = bench.tile_rows(real_rows, config.batch_size) if config.real_logits else \
            bench.create_logits(config.batch_size, config.vocab_size)  # fmt: skip
        for s in args.splits:
            hier_cuda.SPLITS = s
            for row in args.rows:
                trace(config, logits, mhz, row)


def trace(config, logits: torch.Tensor, mhz: float, row: int) -> None:
    rows = logits.shape[0]
    s = hier_cuda.SPLITS or splits(rows, logits.shape[1])
    dbg = torch.zeros(16 + rows, dtype=torch.int64, device="cuda")
    clocks, times = [], []
    for _ in range(5):
        dbg.zero_()
        dbg[15] = row * s
        x = logits.clone()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        apply_top_k_top_p_hier(x, config.k_values, config.p_values, dbg=dbg)
        end.record()
        torch.cuda.synchronize()
        clocks.append(dbg[:15].tolist())
        times.append(1000 * start.elapsed_time(end))
    c = sorted(clocks, key=lambda t: t[10] - t[0])[2]
    steps = [(name, (c[i + 1] - c[i]) / mhz) for i, name in enumerate(PHASES) if c[i + 1] and c[i]]
    paths = dbg[16:].tolist()
    kinds = collections.Counter(f"tier{(x >> 4) & 15}/{['small', 'radix', 'fallback'][x & 15]}" for x in paths)
    counts = sorted(x >> 8 for x in paths if x & 15 != 2)
    median = counts[len(counts) // 2] if counts else None
    fb = f" fallback: hist={(c[8] - c[7]) / mhz:.1f} gather={(c[9] - c[8]) / mhz:.1f} cut={(c[10] - c[9]) / mhz:.1f} generic={(c[13] - max(c[7], c[8], c[10])) / mhz:.1f} bin={c[14]} " if c[13] or c[8] else ""
    sub = fb + f" [slist={(c[11] - c[3]) / mhz:.1f} l1hist={(c[12] - c[11]) / mhz:.1f} glist={(c[4] - c[12]) / mhz:.1f}]"
    print(sub + f"{config.name} row={row} splits={s} event {sorted(times)[2]:.1f}us leader {(c[10] - c[0]) / mhz:.1f}us "
          + " ".join(f"{n}={t:.1f}" for n, t in steps) + f" | {dict(kinds)} median candidates {median}")  # fmt: skip


if __name__ == "__main__":
    main()
