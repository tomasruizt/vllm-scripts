"""Registers, spills and shared memory of the compiled kernels of kernel copies.

Usage: python regs.py base_triton topk_topp_triton
"""

import importlib
import sys

import torch


def main() -> None:
    logits = torch.randn(8, 248320, device="cuda")
    p = torch.full((8,), 0.9, device="cuda")
    for name in sys.argv[1:]:
        mod = importlib.import_module(name)
        mod.apply_top_k_top_p_triton(logits.clone(), None, p)
        for plan in mod._PLANS.values():
            for compiled, *_ in plan:
                print(name, compiled.name, "regs", compiled.n_regs, "spills", compiled.n_spills, "shared", compiled.metadata.shared)


if __name__ == "__main__":
    main()
