"""Real rows, top-p 0.95 only: rows left to the Triton fallback, and kernel resources.

Usage: python triton_core_debug.py --logits-file ROWS.pt
"""

import argparse

import torch
import triton_core


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-file", required=True)
    args = parser.parse_args()
    rows = torch.load(args.logits_file).float().cuda()
    p = torch.full((rows.shape[0],), 0.95, device="cuda")
    triton_core.apply_top_k_top_p_triton_core(rows.clone(), None, p)
    fallback_p = triton_core._BUFFERS[next(iter(triton_core._BUFFERS))][5]
    left = (fallback_p < 1).nonzero().flatten().tolist()
    order = rows.sort(dim=1, descending=True).values
    weight = (order - order[:, :1]).exp()
    nucleus = ((weight.cumsum(1) - weight) < 0.95 * weight.sum(1, keepdim=True)).sum(1)
    print("left to fallback:", left, "nucleus:", nucleus[left].tolist())
    print("nucleus > 128:", (nucleus > 128).nonzero().flatten().tolist())
    for name, kernel in triton_core.LAUNCHED.items():
        print(name, "regs", kernel.n_regs, "spills", kernel.n_spills, "smem", kernel.metadata.shared)


if __name__ == "__main__":
    main()
