"""Cost of the cooperative launch of the PR's filter on ROCm: the same calls with and
without launch_cooperative_grid (the grid fits on the GPU either way), back to back and
between small kernels on the same stream, timed with events (GPU time per iteration).

Usage, in the ROCm image with the PR overlay: python coop_cost.py LABEL ROWS.pt
"""

import sys

import torch

import vllm.v1.sample.ops.topk_topp_triton as hier


def main() -> None:
    source = torch.load(sys.argv[2]).float().cuda()
    for coop in (True, False, True):
        hier.FILTER_OPTS["launch_cooperative_grid"] = coop
        hier._PLANS.clear()
        for rows in (1, 6, 12, 48):
            src = source[torch.arange(rows) % source.shape[0]].contiguous()
            print(sys.argv[1], f"coop={int(coop)} rows={rows}", measure(src), flush=True)


def measure(src: torch.Tensor) -> str:
    logits, x = src.clone(), torch.zeros(1 << 20, device="cuda")
    p = torch.full((src.shape[0],), 0.95, device="cuda")
    filt = lambda: hier.apply_top_k_top_p_triton(logits.copy_(src), None, p)
    small = lambda: [x.add_(1.0) for _ in range(20)]
    out = []
    for name, body in (("copy", lambda: logits.copy_(src)), ("small", small),
                       ("filter", filt), ("small+filter", lambda: (small(), filt()))):
        out.append(f"{name} {gpu_us(body):.1f}")
    return ", ".join(out) + " us/iter"


def gpu_us(body, iters: int = 100) -> float:
    for _ in range(10):
        body()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        body()
    end.record()
    torch.cuda.synchronize()
    return 1e3 * start.elapsed_time(end) / iters


if __name__ == "__main__":
    main()
