"""Timing-only variants of the gather in timed_triton.py (some are inexact), phases on chosen cases.

Usage, from the PR tree: python variants.py --logits-file ROWS.pt --only topp_partial_b256 [--variants base noatomic]
"""

import argparse
import importlib.util
from pathlib import Path

import phases
import torch
from gate import gate_configs
from large_k import load_benchmark

SRC = (Path(__file__).parent / "timed_triton.py").read_text()
ATOMIC = """        pos = tl.atomic_add(
            found_row + offs * 0, 1, mask=hit, sem="relaxed", scope="gpu"
        )
"""
INF = """        tl.store(
            row_ptr + idx, -float("inf"), mask=valid & (b > hi) & (x > -float("inf"))
        )
"""
SCAN = """        hit32 = hit.to(tl.int32)
        n_hit = tl.sum(hit32)
        base = 0
        if n_hit > 0:
            base = tl.atomic_add(found_row, n_hit, sem="relaxed", scope="gpu")
        pos = base + tl.cumsum(hit32, 0) - 1
"""
TAIL = """    tl.atomic_add(more_row, tl.sum(above).to(tl.float64), sem="relaxed", scope="gpu")
"""
MASK_LOOP = """    for start in range(start_at, end, BLOCK):
        idx = start + offs
        x = tl.load(row_ptr + idx, mask=idx < end, other=-float("inf"))
        b = _bin(x, row_max, BIN_SCALE, BINS)
        tl.store(row_ptr + idx, -float("inf"), mask=(idx < end) & (b > hi) & (x > -float("inf")))
"""
COUNT_LOOP = """    n_hit = 0
    for start in range(start_at, end, BLOCK):
        idx = start + offs
        x = tl.load(row_ptr + idx, mask=idx < end, other=-float("inf"))
        b = _bin(x, row_max, BIN_SCALE, BINS)
        n_hit += tl.sum(((idx < end) & (((b >= lo) & (b <= hi)) | (b == kb))).to(tl.int32))
    base = tl.atomic_add(found_row, n_hit, sem="relaxed", scope="gpu") - 1
"""
COUNT_POS = """        hit32 = hit.to(tl.int32)
        pos = base + tl.cumsum(hit32, 0)
        base += tl.sum(hit32)
"""
X_NEXT0 = """    x_next = tl.load(
        row_ptr + start_at + offs, mask=start_at + offs < end, other=-float("inf")
    )
"""

GATHER_CALL = """                    WINDOW,
                    BLOCK,
                )
        _stamp(timing_ptr, pid, 7, TIMING)
"""
OLD_STORES = """        # 32-bit logit and index: the layout conversions of a 64-bit key sync the warps.
        tl.store(
            scratch.to(tl.pointer_type(tl.float32)) + pos, x, mask=hit & (pos < WINDOW)
        )
        tl.store(
            scratch.to(tl.pointer_type(tl.int32)) + WINDOW + pos,
            idx,
            mask=hit & (pos < WINDOW),
        )
"""
PIPE_STORES = """        _put(scratch, x_prev, idx - BLOCK, pos_prev, hit_prev, WINDOW)
        x_prev = x
        pos_prev = pos
        hit_prev = hit
"""
PIPE_INIT = """    x_prev = above
    pos_prev = offs * 0
    hit_prev = offs < 0
    for start in range(start_at, end, BLOCK):
"""
PIPE_TAIL = """    _put(scratch, x_prev, start_at + tl.cdiv(end - start_at, BLOCK) * BLOCK - BLOCK + offs, pos_prev, hit_prev, WINDOW)
    tl.atomic_add(more_row, tl.sum(above).to(tl.float64), sem="relaxed", scope="gpu")
"""
PUT_FN = """@triton.jit
def _put(scratch, x, idx, pos, hit, WINDOW: tl.constexpr):
    tl.store(scratch.to(tl.pointer_type(tl.float32)) + pos, x, mask=hit & (pos < WINDOW))
    tl.store(scratch.to(tl.pointer_type(tl.int32)) + WINDOW + pos, idx, mask=hit & (pos < WINDOW))


@triton.jit(noinline=True)  # one sort per size
"""
VARIANTS = {
    "base": {},
    "noatomic": {ATOMIC: "        pos = (idx - start_at) % WINDOW\n"},
    "noinf": {INF: ""},
    "noboth": {ATOMIC: "        pos = (idx - start_at) % WINDOW\n", INF: ""},
    "scan": {ATOMIC: SCAN},
    "split": {INF: "", TAIL: MASK_LOOP + TAIL},
    "big": {GATHER_CALL: GATHER_CALL.replace("BLOCK,", "2 * BLOCK,")},
    "pipe": {OLD_STORES: PIPE_STORES, TAIL: PIPE_TAIL, "    for start in range(start_at, end, BLOCK):\n": PIPE_INIT,
             "@triton.jit(noinline=True)  # one sort per size\n": PUT_FN},
    "reorder": {INF: "", ATOMIC: INF + ATOMIC},
    "count": {ATOMIC: COUNT_POS, X_NEXT0: COUNT_LOOP + X_NEXT0},
}


def load(name: str):
    src = SRC
    for old, new in VARIANTS[name].items():
        assert src.count(old) == 1, (name, old)
        src = src.replace(old, new)
    path = Path(f"/tmp/work/variant_{name}.py")
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(f"variant_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-file", required=True)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = parser.parse_args()
    bench = load_benchmark(Path("benchmarks/benchmark_topk_topp.py"))
    real_rows = torch.load(args.logits_file).float().cuda()
    configs = [c for c in gate_configs(bench, real_rows) if any(s in c.name for s in args.only)]
    for c in configs:
        logits = bench.tile_rows(real_rows, c.batch_size) if c.real_logits else bench.create_logits(c.batch_size, c.vocab_size)
        for name in args.variants:
            phases.tt = load(name)
            print(c.name, name, phases.phases(logits, c.k_values, c.p_values), flush=True)


if __name__ == "__main__":
    main()
