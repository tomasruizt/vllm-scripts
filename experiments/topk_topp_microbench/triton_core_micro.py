"""Costs of the Triton core prototype's building blocks (sorts, streaming, compaction),
2048 or 16 programs, per-kernel GPU time.

Usage: python triton_core_micro.py
"""

import torch
from torch.profiler import ProfilerActivity, profile

from vllm.triton_utils import tl, triton


@triton.jit
def _sort_kernel(x_ptr, out_ptr, N: tl.constexpr, K: tl.constexpr, MODE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    if MODE == 0:
        y = tl.sort(tl.load(x_ptr + row * N + offs), descending=True)
        tl.store(out_ptr + row * N + offs, y)
    elif MODE == 1:
        y = tl.sort(tl.load(x_ptr + row * N + offs).to(tl.int32, bitcast=True).to(tl.int64))
        tl.store(out_ptr + row * N + offs, y.to(tl.int32).to(tl.float32, bitcast=True))
    else:
        y = tl.topk(tl.load(x_ptr + row * N + offs), K)
        tl.store(out_ptr + row * N + tl.arange(0, K), y)


@triton.jit
def _stream_kernel(x_ptr, out_ptr, count_ptr, vocab, chunk, threshold, BLOCK: tl.constexpr,
                   MODE: tl.constexpr):  # fmt: skip
    row = tl.program_id(0)
    split = tl.program_id(1)
    row_ptr = x_ptr + row.to(tl.int64) * vocab
    lo = split * chunk
    hi = tl.minimum(lo + chunk, vocab)
    offs = tl.arange(0, BLOCK)
    count = 0
    slot = out_ptr + (row * tl.num_programs(1) + split) * 4096
    counter = count_ptr + row * tl.num_programs(1) + split
    if MODE == 3:
        tl.store(counter, 0)
        tl.debug_barrier()
    slots = tl.zeros([BLOCK // 16], tl.int32)
    nxt = tl.load(row_ptr + lo + offs, mask=lo + offs < hi, other=-float("inf"))
    for start in range(lo, hi, BLOCK):
        idx = start + offs
        if MODE == 2:
            x = nxt
            nxt = tl.load(row_ptr + idx + BLOCK, mask=idx + BLOCK < hi, other=-float("inf"))
        else:
            x = tl.load(row_ptr + idx, mask=idx < hi, other=-float("inf"))
        hit = (x > threshold).to(tl.int32)
        if MODE == 3:
            pos = tl.atomic_add(counter + offs * 0, 1, mask=hit > 0, sem="relaxed")
            tl.store(slot + pos, x, mask=(hit > 0) & (pos < 4096))
        elif MODE == 5:
            hit2 = tl.reshape(hit, [BLOCK // 16, 16])
            x2 = tl.reshape(x, [BLOCK // 16, 16])
            pos = slots[:, None] + tl.cumsum(hit2, 1) - 1
            rows2 = tl.arange(0, BLOCK // 16)[:, None]
            tl.store(slot + rows2 * 16 + pos, x2, mask=(hit2 > 0) & (pos < 16))
            slots += tl.sum(hit2, 1)
        elif MODE == 8:
            hits = tl.sum(hit)
            if hits > 0:
                pos = count + tl.cumsum(hit, 0) - 1
                addr = (slot + pos).to(tl.int64, bitcast=True)
                ok = (hit > 0) & (pos < 4096)
                done = tl.inline_asm_elementwise(
                    "{ .reg .pred p; setp.ne.b32 p, $3, 0; @p st.global.b32 [$1], $2; mov.b32 $0, 0; }",
                    "=r,l,r,r", [addr, x.to(tl.int32, bitcast=True), ok.to(tl.int32)],
                    dtype=tl.int32, is_pure=False, pack=1)
                count += tl.max(done)
            count += hits
        elif MODE == 9:
            g = tl.max(tl.reshape(x, [BLOCK // 16, 16]), 1)
            gi = start // 16 + tl.arange(0, BLOCK // 16)
            tl.store(slot + gi, g)
            sg = tl.max(tl.reshape(g, [BLOCK // 256, 16]), 1)
            tl.store(out_ptr + 100_000_000 + row * 1024 + start // 256 + tl.arange(0, BLOCK // 256), sg)
            count += tl.sum(hit)
        elif MODE == 6:
            tl.store(slot + (idx % 4096), x, mask=hit > 0)
        elif MODE == 7:
            count += tl.max(tl.cumsum(hit, 0) * hit)
        elif MODE == 4:
            hit2 = tl.reshape(hit, [BLOCK // 16, 16])
            inner = tl.cumsum(hit2, 1)
            outer = tl.cumsum(tl.sum(hit2, 1), 0) - tl.sum(hit2, 1)
            pos = count + tl.reshape(inner + outer[:, None], [BLOCK]) - 1
            tl.store(slot + pos, x, mask=(hit > 0) & (pos < 4096))
            count += tl.sum(hit)
        else:
            hits = tl.sum(hit)
            if MODE >= 1:
                if hits > 0:
                    pos = count + tl.cumsum(hit, 0) - 1
                    tl.store(slot + pos, x, mask=(hit > 0) & (pos < 4096))
            count += hits
    if MODE == 5:
        tl.store(counter, tl.max(slots))
    elif MODE != 3:
        tl.store(counter, count)


def main() -> None:
    for programs in ():
        x = torch.randn(programs, 4096, device="cuda")
        out = torch.empty_like(x)
        for name, n, k, mode in [("sort f32 4096", 4096, 1, 0), ("sort i64 4096", 4096, 1, 1),
                                 ("sort f32 1024", 1024, 1, 0), ("topk f32 4096->256", 4096, 256, 2),
                                 ("topk f32 4096->1024", 4096, 1024, 2), ("sort f32 256", 256, 1, 0)]:  # fmt: skip
            for warps in (4, 8):
                run = lambda: _sort_kernel[(programs,)](x, out, N=n, K=k, MODE=mode, num_warps=warps)
                print(programs, name, warps, time(run), flush=True)
    vocab = 248320
    for rows, splits in ((2048, 1),):
        logits = torch.randn(rows, vocab, device="cuda")
        out = torch.empty(100_000_000 + rows * 1024, device="cuda")
        counts = torch.empty(rows * splits, dtype=torch.int32, device="cuda")
        chunk = triton.cdiv(triton.cdiv(vocab, splits), 4096) * 4096
        for mode in (0, 1, 8, 9):
            for block, warps in ((4096, 8), (4096, 4), (2048, 4)):
                run = lambda: _stream_kernel[(rows, splits)](logits, out, counts, vocab, chunk, 2.65, BLOCK=block, MODE=mode, num_warps=warps)
                k = run()
                print(k.n_regs, k.n_spills, k.metadata.shared, end=" ")
                print(rows, splits, ["stream", "cumsum", "prefetch+cumsum", "atomic", "2d-scan", "thread-slots", "store-only", "scan-only", "cumsum+asm-store", "group-max"][mode], block, warps, time(run), flush=True)


def time(run) -> float:
    run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            run()
        torch.cuda.synchronize()
    events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return round(sum(e.device_time for e in events) / 10, 1)


if __name__ == "__main__":
    main()
