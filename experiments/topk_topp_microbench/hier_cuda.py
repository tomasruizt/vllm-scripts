"""Hierarchical-maxima top-k/top-p filter in CUDA (hier_cuda.cu), a research kernel
built with torch.utils.cpp_extension. Same contract as apply_top_k_top_p_cuda.

Env: HIER_BUILD_DIR (default ~/code/topk-hier-build), HIER_SPLITS (CTAs per row,
0 = automatic), HIER_CTAS_PER_SM (automatic splits keep rows * splits within
SMs * this, default 2), TORCH_CUDA_ARCH_LIST (default: the current GPU).
For the time breakdown, ablate=1 only streams the rows, ablate=2 streams and fills.
"""

import functools
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

SOURCE = Path(__file__).resolve().parent / "hier_cuda.cu"
SPLITS = int(os.environ.get("HIER_SPLITS", "0"))
CTAS_PER_SM = int(os.environ.get("HIER_CTAS_PER_SM", "2"))


def apply_top_k_top_p_hier(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    ablate: int = 0,
    dbg: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask logits in place outside the per-row top-k, then the top-p nucleus."""
    if k is None and p is None:
        return logits
    k = None if k is None else k.to(torch.int32).contiguous()
    p = None if p is None else p.to(torch.float32).contiguous()
    extension().filter(logits, k, p, SPLITS, CTAS_PER_SM, ablate, dbg)
    return logits


def splits(rows: int, vocab: int) -> int:
    return extension().splits(rows, vocab, CTAS_PER_SM)


@functools.cache
def extension():
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    build = Path(os.environ.get("HIER_BUILD_DIR", "~/code/topk-hier-build")).expanduser()
    build.mkdir(parents=True, exist_ok=True)
    flags = ["-O3", "-lineinfo", "-std=c++20","--expt-relaxed-constexpr"]
    return load("topk_hier", [str(SOURCE)], build_directory=str(build), extra_cuda_cflags=flags)


if __name__ == "__main__":
    extension()  # build only
    print("built", SOURCE.name)
