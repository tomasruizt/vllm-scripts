"""Small CUDA-graph workload for validating NSYS operator-shape capture, no model weights required."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    inputs = [(torch.randn(m, 256, device="cuda", dtype=torch.bfloat16),
               torch.randn(256, n, device="cuda", dtype=torch.bfloat16)) for m, n in ((8, 512), (4, 1024))]
    for a, b in inputs:
        for _ in range(3):
            forward(a, b)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    graphs = []
    outputs = []
    for a, b in inputs:
        graph = torch.cuda.CUDAGraph()
        with region("sol/shape_capture"), torch.cuda.graph(graph):
            outputs.append(forward(a, b))
        graphs.append(graph)
    for step in range(10):
        graph = graphs[step % len(graphs)]
        with region("sol/execute"):
            torch.cuda.nvtx.mark("sol/batch reqs=1 tokens=8 context=0 prefill=0")
            with region("sol/CUDAGraph.replay"):
                graph.replay()
        torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    for (a, b), output in zip(inputs, outputs):
        torch.testing.assert_close(output, forward(a, b))
    info = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "graphs": len(graphs),
            "replays": 10, "dtype": "bf16", "shapes": [[list(a.shape), list(b.shape)] for a, b in inputs]}
    (args.output / "smoke.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info))


def forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.relu(a @ b).sum(dim=-1)


@contextmanager
def region(label: str):
    torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


if __name__ == "__main__":
    main()
