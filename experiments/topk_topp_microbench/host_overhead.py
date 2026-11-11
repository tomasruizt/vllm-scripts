"""Host cost per call of the top-k/top-p masking ops (back-to-back launches, B=1)."""

import time

import torch

from vllm.v1.sample.ops.topk_topp_cuda import apply_top_k_top_p_cuda
from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

IMPLS = {"cuda": apply_top_k_top_p_cuda, "triton": apply_top_k_top_p_triton}


def host_us(func, logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor) -> float:
    for _ in range(100):
        func(logits, k, p)
    torch.cuda.synchronize()
    n = 2000
    start = time.perf_counter()
    for _ in range(n):
        func(logits, k, p)
    elapsed = time.perf_counter() - start
    torch.cuda.synchronize()
    return elapsed / n * 1e6


def main() -> None:
    logits = torch.randn(1, 32768, device="cuda")
    k = torch.full((1,), 50, dtype=torch.int32, device="cuda")
    p = torch.full((1,), 0.9, dtype=torch.float32, device="cuda")
    for name, func in IMPLS.items():
        for label, kk, pp in [("k", k, None), ("p", None, p), ("kp", k, p)]:
            print(f"{name} {label} host={host_us(func, logits, kk, pp):.1f}us")


if __name__ == "__main__":
    main()
