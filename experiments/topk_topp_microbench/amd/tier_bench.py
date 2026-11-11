"""Cost of one selection tier (_tier) per row, alone: rows programs, one row each.

Usage: python tier_bench.py [--rows 8] [--warps 8 16]
"""

import argparse

import torch
import triton
import triton.language as tl
from vllm.v1.sample.ops import topk_topp_triton_hier as hier


@triton.jit
def _tier_kernel(logits_ptr, level1_ptr, level2_ptr, kept_ptr, scratch_ptr, stats_ptr, ok_ptr,
                 vocab, groups, supergroups, k, p, C: tl.constexpr, SUPERGROUPS_P2: tl.constexpr):  # fmt: skip
    row = tl.program_id(0)
    ok, kept = hier._tier(logits_ptr + row.to(tl.int64) * vocab, level1_ptr + row * groups,
                          kept_ptr + row * 256, level2_ptr + row * supergroups,
                          scratch_ptr + row.to(tl.int64) * 4096, supergroups, k, p,
                          tl.load(stats_ptr + 2 * row).to(tl.float32), tl.load(stats_ptr + 2 * row + 1),
                          vocab, groups, C, SUPERGROUPS_P2)  # fmt: skip
    tl.store(ok_ptr + row, tl.where(ok, kept, -1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--warps", type=int, nargs="+", default=[8, 16])
    args = parser.parse_args()
    real = torch.load("qwen36-verify-logits-t1.pt").float().cuda()[: args.rows].contiguous()
    ranks = torch.argsort(torch.rand(args.rows, real.shape[1], device="cuda"), dim=1).float() + 1
    for name, logits in {"real": real, "zipf": -1.1 * ranks.log()}.items():
        for warps in args.warps:
            for c, k, p in [(32, 20, 0.95), (128, real.shape[1], 0.9), (256, 200, 1.0)]:
                bench(name, logits, warps, c, k, p)


def bench(name: str, logits: torch.Tensor, warps: int, c: int, k: int, p: float) -> None:
    rows, vocab = logits.shape
    level1 = logits.view(rows, -1, 16).amax(2).contiguous()
    level2 = logits.view(rows, -1, 256).amax(2).contiguous()
    row_max = logits.max(1, keepdim=True).values
    stats = torch.stack([row_max[:, 0].double(), (logits - row_max).exp().double().sum(1)], 1)
    kept = torch.empty(rows * 256, dtype=torch.int64, device="cuda")
    scratch = torch.empty(rows * 4096, dtype=torch.int64, device="cuda")
    ok = torch.empty(rows, dtype=torch.int32, device="cuda")
    sg = level2.shape[1]
    args = (logits, level1, level2, kept, scratch, stats.flatten().contiguous(), ok, vocab,
            level1.shape[1], sg, k, p, c, max(1024, triton.next_power_of_2(sg)))  # fmt: skip
    compiled = _tier_kernel[(rows,)](*args, num_warps=warps)
    ms = triton.testing.do_bench(lambda: _tier_kernel[(rows,)](*args, num_warps=warps))
    print(f"TIER {name} warps={warps} C={c} k={k} p={p} us={1000 * ms:.1f} kept={ok.tolist()[:4]} "
          f"n_regs={compiled.n_regs} spills={compiled.n_spills}")  # fmt: skip


if __name__ == "__main__":
    main()
