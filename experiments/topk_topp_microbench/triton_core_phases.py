"""Per-phase times of New Triton's persistent kernel (globaltimer stamps) and the row paths,
on the gate's cases (or a name filter of them).

Usage, from the PR tree: python triton_core_phases.py --logits-file ROWS.pt [--only topp_whole k10000]
"""

import argparse
from pathlib import Path

import torch
import triton_core
from gate import gate_configs
from large_k import load_benchmark

# (phase, stamp slot at its end), in time order
PHASES = [("stats", 12), ("tier32", 13), ("tier128", 14), ("select", 1), ("b1", 2), ("hist", 3), ("b2", 4), ("plan", 8), ("b3", 9), ("gather", 10),
          ("b4", 11), ("cut", 5), ("b5", 6), ("fill", 7)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-file", required=True)
    parser.add_argument("--only", nargs="*", default=[], help="substrings of case names")
    args = parser.parse_args()
    bench = load_benchmark(Path("benchmarks/benchmark_topk_topp.py"))
    real_rows = torch.load(args.logits_file).float().cuda()
    for config in gate_configs(bench, real_rows):
        if not args.only or any(s in config.name for s in args.only):
            logits = bench.tile_rows(real_rows, config.batch_size) if config.real_logits else \
                bench.create_logits(config.batch_size, config.vocab_size)  # fmt: skip
            print(config.name, phases(logits, config.k_values, config.p_values), flush=True)


def phases(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> str:
    triton_core.apply_top_k_top_p_triton_core(logits.clone(), k, p)
    triton_core._BUFFERS[(logits.device, *logits.shape)]["found"].fill_(-1)  # -1: not gathered
    triton_core.TIMINGS = torch.zeros(16 * 4096, dtype=torch.int64, device="cuda")
    triton_core.apply_top_k_top_p_triton_core(logits.clone(), k, p)
    torch.cuda.synchronize()
    t = triton_core.TIMINGS.view(-1, 16)
    triton_core.TIMINGS = None
    t = t[t[:, 0] > 0].double() / 1000
    start = t[:, 0].min()
    ends = {n: round((t[:, i].max() - start).item(), 1) for n, i in PHASES if t[:, i].max() > 0}
    b = triton_core._BUFFERS[(logits.device, *logits.shape)]
    paths = {int(v): int(c) for v, c in zip(*b["state"].unique(return_counts=True))}
    found = b["found"].float().quantile(torch.tensor([0.0, 0.5, 1.0], device="cuda")).tolist()
    return f"{len(t)} programs, phase ends µs {ends}, paths {paths}, found min/med/max {found}"


if __name__ == "__main__":
    main()
