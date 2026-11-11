"""Locate the top-p hang by forcing the split count."""

import argparse
import json
import os
from threading import Timer
from pathlib import Path

import torch
import numpy as np

from unified_topk_topp_v3 import UnifiedFilter
from benchmark_unified import oracle_valid

DBG = 32
FIELDS = ["prefix", "above", "candidates", "keep_limit", "k", "above_mass",
          "total_mass_1e6", "bucket"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab", type=int, default=248320)
    parser.add_argument("--reproduce", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.reproduce:
        reproduce(args)
        return
    torch.manual_seed(7)
    logits = torch.randn(4, args.vocab, device="cuda")
    p = torch.tensor([0.8] * 4, device="cuda")
    k20 = torch.tensor([20] * 4, device="cuda", dtype=torch.int32)
    for splits in (1, 8):
        prototype = UnifiedFilter(force_splits=splits)
        for name, k, cutoff in (("topk", k20, None), ("topp", None, p), ("both", k20, p)):
            print(json.dumps(dict(running=name, splits=splits)), flush=True)
            debug = torch.zeros(DBG * logits.shape[0], device="cuda", dtype=torch.int64)
            timer = Timer(30, lambda: os._exit(124))
            timer.daemon = True
            timer.start()
            try:
                out = prototype(logits.clone(), k, cutoff, debug=debug)
                report(name, logits, out)
                valid = oracle_valid(logits, out, k, cutoff)
                state = debug.view(-1, DBG)[0]
                print(json.dumps(dict(case=name, splits=splits, valid=valid,
                                      state=dict(zip(FIELDS, state[:8].tolist())))), flush=True)
                if not valid:
                    raise AssertionError(f"Nucleus contract failed: {name}, splits={splits}")
            finally:
                timer.cancel()


def report(name, source, actual) -> None:
    order = source.argsort(dim=-1, descending=True, stable=True)
    kept = actual.isfinite().gather(-1, order)
    counts = kept.sum(-1).tolist()
    ranks = torch.arange(source.shape[1], device=source.device)
    counted = torch.tensor(counts, device=source.device)
    contiguous = bool(torch.equal(kept, ranks[None, :] < counted[:, None]))
    print(json.dumps(dict(case=name, counts=counts, contiguous=contiguous)), flush=True)


def reproduce(args) -> None:
    from benchmark_unified2 import load_benchmark

    prototype = UnifiedFilter(force_splits=1)
    module = load_benchmark()
    torch.manual_seed(13)
    source = None
    for vocab in [257, 4096, 65536, 248320]:
        for config in module.create_benchmark_configs([1, 4, 128], [vocab], top_k=20, top_p=0.8):
            sample = module.create_logits(config.batch_size, vocab, distribution="normal")
            if vocab == 248320 and config.name == "topp_whole_b128_v248k":
                source = sample
                break
        if source is not None:
            break
    assert source is not None
    p = torch.full((128,), 0.8, device="cuda")
    sorted_source = source.sort(dim=-1, descending=True, stable=True).values.double()
    prefix = sorted_source.softmax(-1).cumsum(-1)
    watchdog = Timer(120, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    try:
        for repeat in range(32):
            debug = torch.zeros((128, DBG), device="cuda", dtype=torch.int64)
            actual = prototype(source.clone(), None, p, debug=debug)
            count = actual.isfinite().sum(-1)
            mass = prefix.gather(-1, (count - 1).clamp_min(0)[:, None]).squeeze(-1)
            previous = prefix.gather(-1, (count - 2).clamp_min(0)[:, None]).squeeze(-1)
            bad = ((mass < p.double() - 1e-7) | (previous > p.double() + 1e-7)).nonzero().flatten().tolist()
            print(json.dumps(dict(repeat=repeat, bad_rows=bad)), flush=True)
            for row in bad[:4]:
                state = debug[row].tolist()
                print(json.dumps(dict(row=row, count=int(count[row]), mass=float(mass[row]),
                                      previous=float(previous[row]), target=float(p[row]),
                                      state=state[:8])), flush=True)
                torch.save(dict(source=source[row].cpu(), actual=actual[row].cpu(),
                                p=float(p[row]), state=state), args.output / f"failure-{row}.pt")
                np.savez(args.output / f"failure-{row}.npz", source=source[row].cpu().numpy(),
                         actual=actual[row].cpu().numpy(), p=float(p[row]), state=state)
    finally:
        watchdog.cancel()


if __name__ == "__main__":
    main()
