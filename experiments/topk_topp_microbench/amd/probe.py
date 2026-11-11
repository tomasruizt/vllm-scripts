"""Compile time, registers and spills of the New Triton kernels, per k/p mix.

Usage, in the ROCm image or on a CUDA host: python probe.py [--rows 64] [--vocab 248320]
"""

import argparse
import time

import torch
from vllm.v1.sample.ops import topk_topp_triton_hier as hier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--vocab", type=int, default=248320)
    args = parser.parse_args()
    logits = torch.randn(args.rows, args.vocab, device="cuda")
    k = torch.full((args.rows,), 20, dtype=torch.int32, device="cuda")
    p = torch.full((args.rows,), 0.95, dtype=torch.float32, device="cuda")
    for name, kk, pp in [("k+p", k, p), ("k", k, None), ("p", None, p)]:
        probe(name, logits, kk, pp)


def probe(name: str, logits: torch.Tensor, k, p) -> None:
    start = time.perf_counter()
    hier.apply_top_k_top_p_triton_hier(logits.clone(), k, p)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    key = next(key for key in hier._PLANS if key[-2:] == (k is not None, p is not None))
    for compiled, grid, _, _ in hier._PLANS[key]:
        meta = compiled.metadata
        print(f"PROBE {name} {compiled.name} grid={grid} n_regs={compiled.n_regs} "
              f"spills={compiled.n_spills} shared={meta.shared} warps={meta.num_warps} "
              f"first_call_s={seconds:.1f}")  # fmt: skip
        functions(compiled.asm.get("amdgcn", ""))
        binary = compiled.asm.get("hsaco") or compiled.asm.get("cubin") or b""
        print(f"  BINARY bytes={len(binary)}")


def functions(asm: str) -> None:
    """Per-function VGPRs and spills of an AMDGCN listing."""
    name = None
    for line in asm.splitlines():
        if line.endswith(":") and not line.startswith((".", "\t", " ")):
            name = line[:-1]
        for field in ("vgpr_count", "vgpr_spill_count", "sgpr_spill_count", "private_segment_fixed_size"):
            if line.strip().startswith(f"; {field}") or f".{field}:" in line:
                print(f"  ASM {name} {line.strip()}")


if __name__ == "__main__":
    main()
