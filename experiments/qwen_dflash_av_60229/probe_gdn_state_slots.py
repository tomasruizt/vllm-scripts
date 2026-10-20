"""Isolate GDN's null-source early return with fixed Qwen3.5 dimensions."""

import json
import os
from pathlib import Path

import torch
from vllm import _custom_ops as ops


@torch.inference_mode()
def main() -> None:
    output = Path(os.environ["PROBE_OUTPUT"])
    rows = []
    for requests in (128, 512):
        for query in (1, 2):
            for null_source in (False, True):
                row = measure(requests, query, null_source)
                rows.append(row)
                print(json.dumps(row), flush=True)
    (output / "gdn-state-slots.json").write_text(json.dumps(rows, indent=2))


def measure(requests: int, query: int, null_source: bool) -> dict:
    tokens = requests * query
    heads = 32
    dimension = 128
    dtype = torch.bfloat16
    mixed = torch.randn(tokens, 8192, device="cuda", dtype=dtype)
    gates = torch.zeros(tokens, heads, device="cuda", dtype=dtype)
    weights = torch.ones(heads, device="cuda", dtype=torch.float32)
    norm = torch.ones(dimension, device="cuda", dtype=dtype)
    indices = torch.zeros(requests, 8, device="cuda", dtype=torch.int32)
    indices[:, :query] = torch.arange(1, tokens + 1, device="cuda", dtype=torch.int32).view(requests, query)
    if null_source:
        indices[:, query - 1] = 0
    offsets = torch.arange(0, tokens + 1, query, device="cuda", dtype=torch.int32)
    accepted = torch.full((requests,), query, device="cuda", dtype=torch.int32)
    state = torch.zeros(tokens + 1, heads, dimension, dimension, device="cuda", dtype=torch.float32)
    output = torch.empty(tokens, heads, dimension, device="cuda", dtype=dtype)
    gate = torch.zeros_like(output)

    def forward() -> None:
        ops.fused_gdn_decode_post_conv_mtp(
            mixed_qkv=mixed, a=gates, b=gates, A_log=weights, dt_bias=weights,
            state_indices=indices, cu_seqlens=offsets, num_accepted_tokens=accepted,
            state=state, output_gate=gate, norm_weight=norm, out=output,
            scale=dimension ** -0.5, norm_eps=1e-6, output_gate_activation="silu",
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        forward()
    times = []
    for _ in range(3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 100)
    assert bool(torch.isfinite(output).all())
    return dict(requests=requests, query=query, null_source=null_source, kernel_ms=times,
                output_abs_max=float(output.abs().max()))


if __name__ == "__main__":
    main()
