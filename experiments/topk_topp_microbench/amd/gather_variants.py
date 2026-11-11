"""Which part of the gather pass is slow on ROCm: variants of _gather_part, alone.

V: 0 as the port, 1 per-hit atomics, 2 no -inf stores, 3 no scatter, 4 loads + sums only.
Usage: python gather_variants.py
"""

import torch
import triton
import triton.language as tl
from vllm.v1.sample.ops import topk_topp_triton_hier as hier


@triton.jit
def _kernel(row_ptr0, scratch0, found_ptr, more_ptr, lo, hi, kb, vocab, chunk, splits,
            BINS: tl.constexpr, WINDOW: tl.constexpr, BLOCK: tl.constexpr, V: tl.constexpr):  # fmt: skip
    row = tl.program_id(0) // splits
    start_at = tl.program_id(0) % splits * chunk
    row_ptr = row_ptr0 + row.to(tl.int64) * vocab
    scratch = scratch0 + row.to(tl.int64) * WINDOW
    found_row = found_ptr + row
    offs = tl.arange(0, BLOCK)
    end = tl.minimum(start_at + chunk, vocab)
    above = tl.zeros([BLOCK], tl.float32)
    between = tl.zeros([BLOCK], tl.float32)
    x_next = tl.load(row_ptr + start_at + offs, mask=start_at + offs < end, other=-float("inf"))
    for start in range(start_at, end, BLOCK):
        idx = start + offs
        valid = idx < end
        x = x_next
        x_next = tl.load(row_ptr + idx + BLOCK, mask=idx + BLOCK < end, other=-float("inf"))
        b = hier._span_bin(x, 0.0, -float("inf"), BINS / 16.0, BINS)
        weight = tl.exp(x)
        above += tl.where(valid & (b < lo), weight, 0.0)
        between += tl.where(valid & (b > hi) & (b < kb), weight, 0.0)
        hit = valid & (((b >= lo) & (b <= hi)) | (b == kb))
        if V != 4:
            if V == 1:
                pos = tl.atomic_add(found_row + offs * 0, 1, mask=hit, sem="relaxed", scope="gpu")
            elif V == 6:
                pos = tl.atomic_add(found_ptr + 64 * row + offs % 64, 1, mask=hit, sem="relaxed", scope="gpu")
            else:
                hits = hit.to(tl.int32)
                pos = tl.cumsum(hits, 0) - 1
                if tl.sum(hits) > 0:
                    pos += tl.atomic_add(found_row, tl.sum(hits), sem="relaxed", scope="gpu")
            if V == 5:
                tl.store(more_ptr + 2 * row, tl.sum(pos).to(tl.float64))
            elif V == 7:
                tl.store(scratch + pos, hier._key(x, idx), mask=hit & (pos < WINDOW))
            elif V == 8:
                tl.store(scratch.to(tl.pointer_type(tl.int32)) + pos, idx, mask=hit & (pos < WINDOW))
            elif V != 3:
                tl.store(scratch.to(tl.pointer_type(tl.float32)) + pos, x, mask=hit & (pos < WINDOW))
                tl.store(scratch.to(tl.pointer_type(tl.int32)) + WINDOW + pos, idx, mask=hit & (pos < WINDOW))
        if V != 2:
            tl.store(row_ptr + idx, -float("inf"), mask=valid & (b > hi) & (x > -float("inf")))
    tl.atomic_add(more_ptr + 2 * row, tl.sum(above).to(tl.float64), sem="relaxed", scope="gpu")
    tl.atomic_add(more_ptr + 2 * row + 1, tl.sum(between).to(tl.float64), sem="relaxed", scope="gpu")


def main() -> None:
    for block in [4096]:
        sweep(256, 4, block)


def sweep(rows: int, splits: int, block: int) -> None:
    vocab, bins = 248320, hier.BINS
    ranks = torch.argsort(torch.rand(rows, vocab, device="cuda"), dim=1).float() + 1
    logits = -1.1 * ranks.log()
    scratch = torch.empty(rows * 4096, dtype=torch.int64, device="cuda")
    found = torch.zeros(rows * 64, dtype=torch.int32, device="cuda")
    more = torch.zeros(rows * 2, dtype=torch.float64, device="cuda")
    chunk = triton.cdiv(triton.cdiv(vocab, splits), block) * block
    for warps in [8]:
        for v in [0, 7, 8, 3]:
            x = logits.clone()
            args = (x, scratch, found, more, 85, 85, bins, vocab, chunk, splits, bins, 4096, block, v)

            def run():
                x.copy_(logits)
                found.zero_()
                _kernel[(rows * splits,)](*args, num_warps=warps)

            compiled = _kernel[(rows * splits,)](*args, num_warps=warps)
            asm = compiled.asm.get("amdgcn", "")
            ops = ["global_store", "global_load", "ds_write", "ds_read", "ds_bpermute", "s_barrier", "s_cbranch", "global_atomic", "buffer_store", "buffer_load"]
            print("ASM", v, {o: asm.count(o) for o in ops})
            if v == -1:
                lines = asm.splitlines()
                at = [i for i, l in enumerate(lines) if "global_store" in l]
                print("\n".join(lines[at[0] - 40 : at[2] + 5]))
            base = triton.testing.do_bench(lambda: x.copy_(logits))
            ms = triton.testing.do_bench(run) - base
            print(f"GV rows={rows} block={block} warps={warps} V={v} hits={found[0].item()} us={1000 * ms:.1f}")


if __name__ == "__main__":
    main()
