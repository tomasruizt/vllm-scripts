"""Kernel-only times (input restored, L2 flushed, then only the call timed) of main's Triton and
the PR kernel, plus per-phase globaltimer stamps of the PR kernel (head_timed.py), on synthetic
top-p cases at V131k and V248k. Prints one JSON line per (case, impl).

Usage: python phases_v131k.py OUT.jsonl [--module head_triton] [--cases topp_whole:2048:131072 ...]
"""

import argparse
import importlib
import json
import statistics
import sys
from pathlib import Path

import torch

CASES = [f"{s}:{b}:{v}" for s in ("topp_whole", "topp_partial") for v in (131072, 248320) for b in (512, 1024, 2048)]
SLOTS = ["select"] + [f"r{r}_{ph}" for r in range(5) for ph in ("hist", "plan", "gather", "cut")]
REPS = 7
WARM = False  # --warm: the input stays in L2 after the restore (a chunk that fits L2)
FLUSH: list[torch.Tensor] = []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--module", default="head_triton", help="kernel copy timed as impl 'new'")
    parser.add_argument("--cases", nargs="+", default=CASES)
    parser.add_argument("--no-main", action="store_true")
    parser.add_argument("--no-phases", action="store_true")
    parser.add_argument("--ref", help="a second kernel copy, timed and checked against 'new'")
    parser.add_argument("--warm", action="store_true")
    args = parser.parse_args()
    global WARM
    WARM = args.warm
    impls = {"new": importlib.import_module(args.module).apply_top_k_top_p_triton}
    if args.ref:
        impls["ref"] = importlib.import_module(args.ref).apply_top_k_top_p_triton
    if not args.no_main:
        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton as main_triton

        impls["triton"] = main_triton
    timed = None if args.no_phases else importlib.import_module(args.module.replace("_triton", "_timed"))
    with args.output.open("a") as out:
        for case in args.cases:
            for line in run_case(case, impls, timed):
                print(json.dumps(line), flush=True)
                out.write(json.dumps(line) + "\n")


def run_case(case: str, impls: dict, timed) -> list[dict]:
    scenario, batch, vocab = case.split(":")
    batch, vocab = int(batch), int(vocab)
    src = create_logits(batch, vocab)
    p = torch.full((batch,), 0.9, device="cuda")
    if scenario == "topp_partial":
        p[batch // 2 :] = 1.0
    x = torch.empty_like(src)
    base = dict(scenario=scenario, batch=batch, vocab=vocab, gb=src.numel() * 4 / 1e9)
    lines = [dict(base, impl=name, us=kernel_time(fn, src, x, p)) for name, fn in impls.items()]
    lines[0]["same_as"] = same_masks(impls, src, x, p)
    if timed is not None:
        lines.append(dict(base, impl="new_phases", **phases(timed, src, x, p)))
    return lines


def same_masks(impls: dict, src, x, p) -> dict:
    """Per impl: whether it keeps exactly the logits that 'new' keeps."""
    masks = {}
    for name, fn in impls.items():
        x.copy_(src)
        masks[name] = torch.isfinite(fn(x, None, p))
    return {name: bool(torch.equal(m, masks["new"])) for name, m in masks.items()}


def kernel_time(fn, src, x, p) -> float:
    times = [one_call(fn, src, x, p) for _ in range(REPS + 1)][1:]
    return round(statistics.median(times), 1)


def one_call(fn, src, x, p) -> float:
    x.copy_(src)
    if not WARM:
        flush_l2()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    fn(x, None, p)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000


def flush_l2() -> None:
    if not FLUSH:
        FLUSH.append(torch.empty(512 << 20, dtype=torch.int8, device="cuda"))
    FLUSH[0].zero_()


def phases(tt, src, x, p) -> dict:
    runs = [phase_run(tt, src, x, p) for _ in range(REPS + 1)][1:]
    runs.sort(key=lambda r: r["total"])
    return runs[len(runs) // 2]


def phase_run(tt, src, x, p) -> dict:
    stamps = torch.zeros(2 + 4096 * 32, dtype=torch.int64, device="cuda")
    stamps[0] = torch.iinfo(torch.int64).max
    tt.TIMINGS = stamps
    total = one_call(tt.apply_top_k_top_p_triton, src, x, p)
    tt.TIMINGS = None
    return dict(total=round(total, 1), **durations(stamps))


def durations(stamps: torch.Tensor) -> dict:
    """Per phase: from the previous barrier's last arrival to this one's last stamp (µs)."""
    lv0, lv1 = stamps[0].item(), stamps[1].item()
    t = stamps[2:].view(-1, 32)
    t = t[t[:, 0] > 0].double()
    out = {"programs": t.shape[0], "levels": round((lv1 - lv0) / 1000, 1), "gap": round((t[:, 0].min().item() - lv1) / 1000, 1)}
    prev = t[:, 0].min().item()
    for i, name in enumerate(SLOTS + ["fill"], start=1):
        col = t[:, 31 if name == "fill" else i]
        if col.max() > 0:
            out[name] = round((col.max().item() - prev) / 1000, 1)
            prev = col.max().item()
    out["filter"] = round((prev - t[:, 0].min().item()) / 1000, 1)
    return out



def create_logits(batch: int, vocab: int) -> torch.Tensor:
    """The benchmark's synthetic rows: Zipf (rank^-1.1) log-probs, permuted per row."""
    probs = torch.arange(1, vocab + 1, dtype=torch.float32, device="cuda").pow(-1.1)
    base = (probs / probs.sum()).log()
    return torch.stack([base[torch.randperm(vocab, device="cuda")] for _ in range(batch)])

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
