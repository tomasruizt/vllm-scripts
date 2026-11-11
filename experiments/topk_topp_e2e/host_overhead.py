"""Host cost of one top-p filter call on the serving path (apply_top_k_top_p), and whether
the call blocks the host while the GPU is busy (an implicit sync defeats async scheduling).

Usage, in the ROCm image (amd_e2e.sh with RUN): python host_overhead.py LABEL [ROWS.pt]
"""

import sys
import time

import torch

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

VOCAB = 154880


def main() -> None:
    label = sys.argv[1]
    source = load_rows(sys.argv[2] if len(sys.argv) > 2 else None)
    busy = busy_work()
    for rows in (1, 6, 12, 48):
        src = source[torch.arange(rows) % source.shape[0]].contiguous()
        print(label, rows, measure(src, busy), flush=True)


def load_rows(path: str | None) -> torch.Tensor:
    if path is None:
        return torch.randn(64, VOCAB, device="cuda") * 4
    return torch.load(path).float().cuda()


def busy_work():
    """~20 ms of GPU work to queue ahead of a call."""
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    def run() -> None:
        for _ in range(20):
            a @ a
    run(); torch.cuda.synchronize()
    t = time.perf_counter(); run(); torch.cuda.synchronize()
    print(f"busy work {1e3 * (time.perf_counter() - t):.1f} ms", flush=True)
    return run


def measure(src: torch.Tensor, busy) -> str:
    logits = src.clone()
    p = torch.full((src.shape[0],), 0.95, device="cuda")
    call = lambda: apply_top_k_top_p(logits.copy_(src), None, p)
    for _ in range(20):
        call()
    torch.cuda.synchronize()
    iters = 200
    t0 = time.perf_counter()
    for _ in range(iters):
        call()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    host, total = 1e6 * (t1 - t0) / iters, 1e6 * (t2 - t0) / iters
    blocked = []
    for _ in range(5):
        busy()
        t3 = time.perf_counter()
        call()
        blocked.append(1e6 * (time.perf_counter() - t3))
        torch.cuda.synchronize()
    return f"host {host:.1f} us/call, total {total:.1f} us/call, call behind busy GPU {sorted(blocked)[2]:.1f} us"


if __name__ == "__main__":
    main()
