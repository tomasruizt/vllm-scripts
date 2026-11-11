"""Per-phase times (globaltimer stamps) of the filter kernel in timed_triton.py, on gate cases.

Usage, from the PR tree: python phases.py --logits-file ROWS.pt [--only topp_partial_b256]
"""

import argparse
from pathlib import Path

import timed_triton as tt
import torch
from gate import gate_configs
from large_k import load_benchmark

NAMES = ["select", "b1", "hist", "b2", "plan", "b3", "gather", "b4", "cut", "b5", "fill"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-file", required=True)
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()
    bench = load_benchmark(Path("benchmarks/benchmark_topk_topp.py"))
    real_rows = torch.load(args.logits_file).float().cuda()
    for c in gate_configs(bench, real_rows):
        if args.only and not any(s in c.name for s in args.only):
            continue
        logits = bench.tile_rows(real_rows, c.batch_size) if c.real_logits else bench.create_logits(c.batch_size, c.vocab_size)
        print(c.name, phases(logits, c.k_values, c.p_values), flush=True)


def phases(logits: torch.Tensor, k, p) -> str:
    tt.apply_top_k_top_p_triton(logits.clone(), k, p)
    runs = [one(logits, k, p) for _ in range(5)]
    span, durs, total = runs[len(runs) // 2]
    b = tt._BUFFERS[(logits.device, logits.shape[1])]
    paths = {int(v): int(n) for v, n in zip(*b["state"][: logits.shape[0]].unique(return_counts=True))}
    return f"total {total:.1f} filter {span:.1f} µs, phases {durs}, paths {paths}"


FLUSH = []  # an L2-sized buffer zeroed before each call, as the gate's cold L2


def one(logits, k, p):
    if not FLUSH:
        FLUSH.append(torch.empty(256 << 20, dtype=torch.int8, device="cuda"))
    FLUSH[0].zero_()
    x = logits.clone()
    tt.TIMINGS = torch.zeros(16 * 4096, dtype=torch.int64, device="cuda")
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    tt.apply_top_k_top_p_triton(x, k, p)
    end.record()
    torch.cuda.synchronize()
    t = tt.TIMINGS.view(-1, 16)
    tt.TIMINGS = None
    t = t[t[:, 0] > 0].double() / 1000
    t0 = t[:, 0].min()
    ends = [(t[:, i].max() - t0).item() if t[:, i].max() > 0 else None for i in range(1, 12)]
    durs, prev = {}, 0.0
    for name, e in zip(NAMES, ends):
        if e is not None:
            durs[name] = round(e - prev, 1)
            prev = e
    return prev, durs, start.elapsed_time(end) * 1000


if __name__ == "__main__":
    main()
