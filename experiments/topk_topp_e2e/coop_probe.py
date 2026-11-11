"""Does the PR's cooperative filter launch slow down the kernels that follow it on ROCm?
Each iteration calls the filter (6 real GLM rows), then replays a HIP graph of small
kernels and weight-streaming GEMMs (as in a decode step), and times only the graph.

Usage, in the ROCm image with the PR overlay: python coop_probe.py LABEL ROWS.pt MODE
MODE: coop (as in the PR), coop-sync (host sync after the filter), plain (same kernel,
launch_cooperative_grid=False), none.
"""

import sys

import torch

import vllm.v1.sample.ops.topk_topp_triton as hier


def main() -> None:
    label, rows_path, mode = sys.argv[1:4]
    hier.FILTER_OPTS["launch_cooperative_grid"] = mode.startswith("coop")
    src = torch.load(rows_path).float().cuda()[:6].contiguous()
    logits, p = src.clone(), torch.full((6,), 0.95, device="cuda")
    call = lambda: hier.apply_top_k_top_p_triton(logits.copy_(src), None, p)
    filt = {"none": lambda: None, "coop-sync": lambda: (call(), torch.cuda.synchronize())}.get(mode, call)
    for name, graph in (("small", graph_of(small)), ("gemm", graph_of(gemm)), ("eager", eager)):
        print(f"{label} mode={mode} {name}: {graph_us(graph, filt):.1f} us per replay", flush=True)


x = torch.zeros(1 << 16, device="cuda")
h = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
ws = [torch.randn(4096, 6144, device="cuda", dtype=torch.bfloat16) for _ in range(8)]  # 400 MB


def small() -> None:
    for _ in range(250):
        x.add_(1.0)
        x.mul_(0.5)
        x.sqrt_()
        h @ ws[0][:, :512]


def gemm() -> None:
    for w in ws:
        h @ w


class eager:
    """Eager launches of small kernels, in place of a graph."""

    @staticmethod
    def replay() -> None:
        for _ in range(50):
            x.add_(1.0)


def graph_of(body) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            body()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    return graph


def graph_us(graph: torch.cuda.CUDAGraph, filt, iters: int = 50) -> float:
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for _ in range(5):
        filt()
        graph.replay()
    torch.cuda.synchronize()
    for start, end in events:
        filt()
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    times = sorted(1e3 * s.elapsed_time(e) for s, e in events)
    return times[len(times) // 2]


if __name__ == "__main__":
    main()
