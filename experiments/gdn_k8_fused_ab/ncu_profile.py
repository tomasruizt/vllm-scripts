"""Eager launches of the fused GDN MTP decode kernel for one shape, for NCU (same inputs as kernel_bench.py)."""

import argparse

import torch

import kernel_bench as kb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--state-dtype", default="float32")
    parser.add_argument("--launches", type=int, default=8, help="each launch uses its own state slots")
    args = parser.parse_args()
    x = kb.make_inputs(args.batch, args.tokens, getattr(torch, args.state_dtype))
    for indices in x["indices"][: args.launches]:
        kb.run_cuda(x, indices)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
