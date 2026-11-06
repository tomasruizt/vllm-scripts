"""GDN core: sequential recurrent kernel vs chunked prefill kernels, as a function of tokens per request.

Shapes are Qwen3.6-35B-A3B at TP1 (H=16, HV=32, K=V=128), fp32 state pool, q/k already L2-normalized.
Only the recurrence is timed (no conv, gating prep, norm or projections). Variants:
  recurrent_all    Triton fused_recurrent_gated_delta_rule, stores the state after every token (spec decode semantics)
  recurrent_final  same kernel, stores only the final state (prefill semantics)
  chunk_fi         FlashInfer chunk_gated_delta_rule (vLLM's default prefill on SM90/SM100), incl. state gather/scatter
  chunk_fla        FLA Triton chunk_gated_delta_rule (vLLM's fallback prefill), incl. state gather/scatter
Each timed call uses its own state slots (state read from HBM). CUDA graph replay when capturable, else eager.
"""

import argparse
import json

import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import fi_chunk_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import fused_recurrent_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices, prepare_chunk_offsets

H, HV, K, V = 16, 32, 128, 128
ITERS = 10
STATE_BUDGET_BYTES = 8 << 30
MAX_TOKENS = 65536


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 4, 8, 9, 16, 32, 64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--variants", nargs="+", default=["recurrent_all", "recurrent_final", "chunk_fi", "chunk_fla"])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    gpu = torch.cuda.get_device_name()
    for requests in args.requests:
        for length in args.lengths:
            if requests * length > MAX_TOKENS:
                continue
            x = make_inputs(requests, length)
            for repeat in range(args.repeats):
                for variant in args.variants:
                    us, mode = time_us(x, variant)
                    print(json.dumps(dict(gpu=gpu, variant=variant, requests=requests, length=length,
                                          repeat=repeat, us=us, mode=mode)), flush=True)


def make_inputs(requests: int, length: int) -> dict:
    torch.manual_seed(0)
    device, n = "cuda", requests * length
    slots_per_iter = requests * length
    num_slots = min(ITERS * slots_per_iter, STATE_BUDGET_BYTES // (HV * V * K * 4)) + 1
    q = torch.nn.functional.normalize(torch.randn(1, n, H, K, device=device), dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(torch.randn(1, n, H, K, device=device), dim=-1).to(torch.bfloat16)
    v = torch.randn(1, n, HV, V, dtype=torch.bfloat16, device=device)
    g = torch.nn.functional.logsigmoid(torch.rand(1, n, HV, device=device) + 2)
    beta = torch.rand(1, n, HV, device=device).sigmoid()
    state = 0.01 * torch.randn(num_slots, HV, V, K, device=device)
    all_slots = [state_slots(i, requests, length, num_slots, device) for i in range(ITERS)]
    cu_seqlens = torch.arange(0, n + 1, length, dtype=torch.int32, device=device)
    final_only = {slots.data_ptr(): final_only_slots(slots) for slots in all_slots}
    return dict(q=q, k=k, v=v, g=g, beta=beta, state=state, all_slots=all_slots, cu_seqlens=cu_seqlens, final_only=final_only,
                chunk_indices=prepare_chunk_indices(cu_seqlens, 64), chunk_offsets=prepare_chunk_offsets(cu_seqlens, 64))


def state_slots(i: int, requests: int, length: int, num_slots: int, device: str) -> torch.Tensor:
    first = i * requests * length
    return (1 + (torch.arange(requests * length, device=device) + first) % (num_slots - 1)).to(torch.int32).view(
        requests, length)


def final_only_slots(slots: torch.Tensor) -> torch.Tensor:
    """Slot indices that make the recurrent kernel store only the final state (column 0 is the initial state)."""
    final_only = torch.zeros_like(slots)
    final_only[:, 0] = slots[:, 0]
    final_only[:, -1] = slots[:, 0]
    return final_only


def time_us(x: dict, variant: str) -> tuple[float, str]:
    step = STEPS[variant]
    for slots in x["all_slots"][:2]:
        step(x, slots)
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for slots in x["all_slots"]:
                step(x, slots)
        run, mode = graph.replay, "graph"
    except Exception:
        torch.cuda.synchronize()
        run, mode = (lambda: [step(x, slots) for slots in x["all_slots"]]), "eager"
    run()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        run()
    end.record()
    torch.cuda.synchronize()
    return round(start.elapsed_time(end) * 1e3 / (10 * ITERS), 2), mode


def recurrent_all(x: dict, slots: torch.Tensor) -> None:
    fused_recurrent_gated_delta_rule(q=x["q"], k=x["k"], v=x["v"], g=x["g"], beta=x["beta"], initial_state=x["state"],
                                     inplace_final_state=True, cu_seqlens=x["cu_seqlens"], ssm_state_indices=slots)


def recurrent_final(x: dict, slots: torch.Tensor) -> None:
    final_only = x["final_only"][slots.data_ptr()]
    fused_recurrent_gated_delta_rule(q=x["q"], k=x["k"], v=x["v"], g=x["g"], beta=x["beta"], initial_state=x["state"],
                                     inplace_final_state=True, cu_seqlens=x["cu_seqlens"], ssm_state_indices=final_only)


def chunk_fi(x: dict, slots: torch.Tensor) -> None:
    index = slots[:, 0]
    _, final = fi_chunk_gated_delta_rule(q=x["q"], k=x["k"], v=x["v"], g=x["g"].exp(), beta=x["beta"],
                                         initial_state=x["state"][index], output_final_state=True,
                                         cu_seqlens=x["cu_seqlens"], use_qk_l2norm_in_kernel=False)
    x["state"][index] = final


def chunk_fla(x: dict, slots: torch.Tensor) -> None:
    index = slots[:, 0]
    _, final = chunk_gated_delta_rule(q=x["q"], k=x["k"], v=x["v"], g=x["g"], beta=x["beta"],
                                      initial_state=x["state"][index], output_final_state=True,
                                      cu_seqlens=x["cu_seqlens"], chunk_indices=x["chunk_indices"],
                                      chunk_offsets=x["chunk_offsets"])
    x["state"][index] = final


STEPS = dict(recurrent_all=recurrent_all, recurrent_final=recurrent_final, chunk_fi=chunk_fi, chunk_fla=chunk_fla)

if __name__ == "__main__":
    main()
