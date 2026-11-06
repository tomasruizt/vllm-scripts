"""GDN MTP decode microbenchmark: fused CUDA kernel of the vLLM on PYTHONPATH vs Triton delta-rule + RMSNorm.

Shapes are Qwen3.6-35B-A3B at TP `--tp` (H=16/tp, HV=32/tp, K=V=128) with `--tokens` per request (9 for DSpark K=8, 4 for MTP3).
`--splits` sets VLLM_GDN_DECODE_SPLIT_V, which only a tuning build of the split-V kernel reads ("auto" unsets it).
Each timed iteration uses its own state slots, so the state comes from HBM (other layers evict L2 in serving).
Times are CUDA-graph replays of `ITERS` calls, reported per call. Output: one JSON line per cell.
"""

import argparse
import json
import os

import torch

from vllm import _custom_ops as ops
from vllm.third_party.flash_linear_attention.ops import fused_sigmoid_gating_delta_rule_update
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn

H, HV, K, V = 16, 32, 128, 128
ITERS = 20
STATE_BUDGET_BYTES = 6 << 30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--tokens", type=int, default=9)
    parser.add_argument("--state-dtypes", nargs="+", default=["float32", "bfloat16"])
    parser.add_argument("--label", required=True, help="Source variant of the CUDA kernel, e.g. pr or splitv")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--impls", nargs="+", default=["triton", "cuda"])
    parser.add_argument("--splits", nargs="+", default=["auto"])
    args = parser.parse_args()
    global H, HV
    H, HV = H // args.tp, HV // args.tp
    gpu = torch.cuda.get_device_name()
    cells = [(impl, split) for impl in args.impls for split in (args.splits if impl == "cuda" else ["auto"])]
    for dtype in args.state_dtypes:
        for batch in args.batches:
            inputs = make_inputs(batch, args.tokens, getattr(torch, dtype))
            for repeat in range(args.repeats):
                for impl, split in cells:
                    us = time_us(inputs, impl, split)
                    print(json.dumps(dict(label=args.label, gpu=gpu, tp=args.tp, hv=HV, impl=impl, split=split,
                                          repeat=repeat, state_dtype=dtype, batch=batch, tokens=args.tokens, us=us)),
                          flush=True)


def make_inputs(batch: int, tokens: int, state_dtype: torch.dtype) -> dict:
    torch.manual_seed(0)
    device = "cuda"
    num_tokens = batch * tokens
    slot_bytes = HV * V * K * state_dtype.itemsize
    slots_per_iter = batch * tokens
    num_slots = min(ITERS * slots_per_iter, STATE_BUDGET_BYTES // slot_bytes) + 1
    mixed_qkv = torch.randn(num_tokens, 2 * H * K + HV * V, dtype=torch.bfloat16, device=device)
    ba = torch.randn(num_tokens, 2 * HV, dtype=torch.bfloat16, device=device)
    b, a = ba.chunk(2, dim=-1)
    state = (0.01 * torch.randn(num_slots, HV, V, K, device=device)).to(state_dtype)
    indices = [(1 + (torch.arange(slots_per_iter, device=device) + i * slots_per_iter) % (num_slots - 1))
               .to(torch.int32).view(batch, tokens) for i in range(ITERS)]
    return dict(mixed_qkv=mixed_qkv, a=a, b=b, state=state, indices=indices,
                A_log=0.5 * torch.randn(HV, device=device), dt_bias=0.1 * torch.randn(HV, device=device),
                output_gate=torch.randn(num_tokens, HV, V, dtype=torch.bfloat16, device=device),
                norm_weight=torch.randn(V, dtype=torch.bfloat16, device=device),
                out=torch.empty(num_tokens, HV, V, dtype=torch.bfloat16, device=device),
                cu_seqlens=torch.arange(0, num_tokens + 1, tokens, dtype=torch.int32, device=device),
                accepted=torch.full((batch,), 2, dtype=torch.int32, device=device))


def time_us(x: dict, impl: str, split: str) -> float:
    os.environ.pop("VLLM_GDN_DECODE_SPLIT_V", None)
    if split != "auto":
        os.environ["VLLM_GDN_DECODE_SPLIT_V"] = split
    step = run_cuda if impl == "cuda" else run_triton
    for indices in x["indices"][:3]:
        step(x, indices)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for indices in x["indices"]:
            step(x, indices)
    graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return round(start.elapsed_time(end) * 1e3 / (20 * ITERS), 2)


def run_cuda(x: dict, indices: torch.Tensor) -> None:
    ops.fused_gdn_decode_post_conv_mtp(
        mixed_qkv=x["mixed_qkv"], a=x["a"], b=x["b"], A_log=x["A_log"], dt_bias=x["dt_bias"],
        state_indices=indices, cu_seqlens=x["cu_seqlens"], num_accepted_tokens=x["accepted"],
        state=x["state"], output_gate=x["output_gate"], norm_weight=x["norm_weight"], out=x["out"],
        scale=K**-0.5, norm_eps=1e-6, output_gate_activation="silu")


def run_triton(x: dict, indices: torch.Tensor) -> None:
    n = x["mixed_qkv"].shape[0]
    q, k, v = torch.split(x["mixed_qkv"], [H * K, H * K, HV * V], dim=-1)
    raw, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=x["A_log"], a=x["a"], b=x["b"], dt_bias=x["dt_bias"],
        q=q.view(1, n, H, K), k=k.view(1, n, H, K), v=v.view(1, n, HV, V),
        initial_state=x["state"], inplace_final_state=True, cu_seqlens=x["cu_seqlens"],
        ssm_state_indices=indices, num_accepted_tokens=x["accepted"], scale=K**-0.5,
        use_qk_l2norm_in_kernel=True)
    rmsnorm_fn(raw.squeeze(0), x["norm_weight"], None, z=x["output_gate"], eps=1e-6,
               norm_before_gate=True, activation="silu")


if __name__ == "__main__":
    main()
