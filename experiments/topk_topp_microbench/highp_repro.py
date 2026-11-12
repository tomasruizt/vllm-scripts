"""Top-p-only rows at high p (reported on gfx942: 9 to 18 ms per call): kernel time of the
new kernel vs. main's Triton kernel on real verify rows, and the share of rows whose
nucleus reaches beyond the 16 nats below the row max (the histogram span's open last bin).

Usage: highp_repro.py OUT.jsonl --rows NAME=PATH ... [--batches 8 64 512] [--ps 0.9 0.99]
    [--modules fix ...]
Needs main's kernel as triton_kernel.py next to this script. Each --modules NAME adds
NAME_kernel.py (a variant of the new kernel) under NAME.
"""

import argparse
import importlib
import json
from pathlib import Path

import torch
import triton_kernel

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

SPAN_NATS = 16.0  # the first round's histogram span below the row max
IMPLS = {"new": apply_top_k_top_p_triton, "main": triton_kernel.apply_top_k_top_p_triton}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--rows", nargs="+", required=True, help="NAME=PATH of [N, V] FP32 logits")
    parser.add_argument("--batches", type=int, nargs="+", default=[8, 64, 512])
    parser.add_argument("--ps", type=float, nargs="+", default=[0.9, 0.95, 0.98, 0.99, 0.995, 0.999])
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--modules", nargs="*", default=[])
    args = parser.parse_args()
    for name in args.modules:
        IMPLS[name] = importlib.import_module(f"{name}_kernel").apply_top_k_top_p_triton
    with args.out.open("w") as out:
        for name, path in (spec.split("=") for spec in args.rows):
            rows = torch.load(path).float().cuda()
            for p in args.ps:
                beyond = beyond_span(rows, p)
                for batch in args.batches:
                    row = measure(rows, batch, p, args.iters) | dict(source=name, p=p, batch=batch, beyond_span=beyond)
                    print(json.dumps(row), flush=True)
                    out.write(json.dumps(row) + "\n")


def beyond_span(rows: torch.Tensor, p: float) -> float:
    """Share of rows whose top-p nucleus needs logits more than SPAN_NATS below the max."""
    shifted = rows - rows.max(-1, keepdim=True).values
    weight = shifted.exp()
    inside = (weight * (shifted >= -SPAN_NATS)).sum(-1) / weight.sum(-1)
    return (inside < p).float().mean().item()


def measure(rows: torch.Tensor, batch: int, p: float, iters: int) -> dict:
    logits = rows.repeat((batch + len(rows) - 1) // len(rows), 1)[:batch].contiguous()
    ps = torch.full((batch,), p, device="cuda")
    reference = apply_top_k_top_p_pytorch(logits.clone(), None, ps) > -float("inf")
    result, kept_new = {}, None
    for name, func in IMPLS.items():
        kept = func(logits.clone(), None, ps) > -float("inf")
        kept_new = kept if kept_new is None else kept_new
        result[f"{name}_exact"] = bool(torch.equal(kept, reference))
        result[f"{name}_valid"] = valid_top_p(logits, kept, p)
        result[f"{name}_same_as_new"] = bool(torch.equal(kept, kept_new))
        result[f"{name}_us"] = time_us(func, logits, ps, iters)
    return result


def valid_top_p(logits: torch.Tensor, kept: torch.Tensor, p: float, tol: float = 1e-7) -> bool:
    """Every row keeps a prefix of the stable order (logit desc, index asc) whose FP64
    mass reaches p, without its last logit being needed (both within tol)."""
    x = logits.double()
    values, order = torch.sort(x, dim=-1, descending=True, stable=True)
    probs = torch.softmax(values, dim=-1)
    cum = probs.cumsum(-1)
    count = kept.sum(-1)
    prefix = torch.arange(x.shape[-1], device=x.device) < count[:, None]
    is_prefix = torch.equal(torch.gather(kept, 1, order), prefix)
    last = (count - 1).clamp(min=0)[:, None]
    reach = cum.gather(1, last).squeeze(1)
    before = reach - probs.gather(1, last).squeeze(1)
    return bool(is_prefix and (reach >= p - tol).all() and (before < p + tol).all())


def time_us(func, logits: torch.Tensor, ps: torch.Tensor, iters: int) -> float:
    """Median per call, each on a fresh copy of the input with the caches flushed (untimed)."""
    work, flush = torch.empty_like(logits), torch.empty(1 << 28, device="cuda")
    times = []
    for i in range(iters + 2):
        work.copy_(logits)
        flush.zero_()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        func(work, None, ps)
        end.record()
        torch.cuda.synchronize()
        if i >= 2:
            times.append(start.elapsed_time(end) * 1000)
    return sorted(times)[len(times) // 2]


if __name__ == "__main__":
    main()
