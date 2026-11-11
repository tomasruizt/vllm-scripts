"""Measure the per-pass cost floor of the primitives a unified filter needs."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

MODES = ["read", "hist", "hist_mass", "hist_f32", "exp_sum", "mask",
         "hist_refine", "mask_refine"]
MODE_IDS = {name: index for index, name in enumerate(MODES)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", default="1,64,512")
    parser.add_argument("--vocabs", default="65536,248320")
    parser.add_argument("--splits", default="1,4,16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    module = load(
        name="filter_passes",
        sources=[str(Path(__file__).with_name("filter_passes.cu"))],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        verbose=False,
    )
    rows = sweep(module, args)
    (args.output / "passes.json").write_text(json.dumps(rows, indent=2))
    print_table(rows)
    (args.output / "DONE").write_text("ok\n")


def sweep(module, args) -> list[dict]:
    batches = [int(x) for x in args.batches.split(",")]
    vocabs = [int(x) for x in args.vocabs.split(",")]
    splits = [int(x) for x in args.splits.split(",")]
    rows = []
    for vocab in vocabs:
        for batch in batches:
            logits = torch.randn(batch, vocab, device="cuda", dtype=torch.float32)
            out = torch.empty_like(logits)
            ref, prefix, prefix_mask = refine_params(logits)
            for split in splits:
                for name in MODES:
                    duration = time_pass(module, logits, out, name, split,
                                         ref, prefix, prefix_mask, args)
                    rows.append(dict(mode=name, batch=batch, vocab=vocab,
                                     splits=split, us=duration))
                    print(json.dumps(rows[-1]), flush=True)
    return rows


def refine_params(logits) -> tuple[float, int, int]:
    """Threshold keeping roughly 1000 elements per row, as an 8-bit prefix."""
    value = float(torch.quantile(logits[0], 1 - 1000 / logits.shape[1]))
    bits = int(torch.tensor(value, dtype=torch.float32).view(torch.int32))
    bits &= 0xFFFFFFFF
    key = (~bits & 0xFFFFFFFF) if bits & 0x80000000 else bits ^ 0x80000000
    return value, key & 0xFF000000, 0xFF000000


def time_pass(module, logits, out, name, split, ref, prefix, prefix_mask, args) -> float:
    run = lambda: module.run(logits, out, MODE_IDS[name], ref, prefix, prefix_mask, split)
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(args.iters):
        run()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / args.iters


def print_table(rows: list[dict]) -> None:
    header = f"{'mode':12s} {'vocab':>7s} {'batch':>5s} {'splits':>6s} {'us':>9s} {'ns/1k':>8s}"
    print(header)
    for row in rows:
        per_1k = row["us"] * 1e3 / (row["batch"] * row["vocab"] / 1000)
        print(f"{row['mode']:12s} {row['vocab']:7d} {row['batch']:5d} {row['splits']:6d} "
              f"{row['us']:9.2f} {per_1k:8.3f}")


if __name__ == "__main__":
    main()
