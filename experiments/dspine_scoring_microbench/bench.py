"""Microbenchmark of DSpine transition scoring (vLLM PR #67094, head c1ef7580c6).

Compares, at Qwen3.8-27B-DSpine-b16 shapes and per batch size:
- `ref`: the PR's `_score_transitions` (TP=1 path), copied verbatim.
- `ref_compiled`: the same function under `torch.compile`.
- `lowrank`: the same math in the 512-dim message space, with per-vocab tables.
- `fused`: `lowrank` with a Triton per-position kernel and a TF32 `u^T G u` mm.
- `lmhead_topk`: the draft's full-vocab lm_head + top-k (`compute_candidates`), a
  yardstick for the inherent per-step draft cost.

Every variant runs inside a CUDA graph (as in vLLM's FULL draft graphs). Random
weights: timings are representative, score errors only indicative.
"""

import argparse
import csv
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

H, T, M, K, S, V, L = 5120, 1024, 512, 16, 15, 248320, 5
EPS, NORM_EPS = 1e-6, 1e-6
DTYPE = torch.bfloat16


def main() -> None:
    global VARIANTS
    args = parse_args()
    VARIANTS = args.variants.split(",")
    torch.manual_seed(0)
    weights = make_weights()
    tables = make_tables(weights)
    rows = []
    for batch in args.batches:
        rows += bench_batch(weights, tables, batch, args.iters, args.precision)
    write_csv(args.out, rows)
    if args.profile_batch:
        profile_variant(weights, tables, args.profile_variant, args.profile_batch)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 64, 128, 256])
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--profile-batch", type=int, default=256)
    p.add_argument("--profile-variant", default="ref")
    p.add_argument("--out", default="results.csv")
    # fp32 matmul precision of the variants ("high" = TF32 tensor cores). The
    # reference scores for the errors are always computed at "highest".
    p.add_argument("--precision", default="highest", choices=["highest", "high"])
    p.add_argument("--variants", default="ref,ref_compiled,lowrank,fused,lmhead_topk")
    return p.parse_args()


def bench_batch(
    w: "Weights", tables: "Tables", batch: int, iters: int, precision: str
) -> list[dict]:
    inputs = make_inputs(batch)
    variants: dict[str, Callable[[], torch.Tensor]] = {
        "ref": lambda: score_ref(w, *inputs),
        "ref_compiled": lambda: score_ref_compiled(w, *inputs),
        "lowrank": lambda: score_lowrank(w, tables, *inputs),
        "fused": lambda: score_fused(w, tables, *inputs),
        "fused2": lambda: score_fused2(w, tables, *inputs),
        "fused3": lambda: score_fused3(w, tables, *inputs),
        "lmhead_topk": lambda: lmhead_topk(w, batch),
    }
    variants = {k: v for k, v in variants.items() if k in VARIANTS}
    torch.set_float32_matmul_precision("highest")
    ref_scores = score_ref(w, *inputs)
    torch.set_float32_matmul_precision(precision)
    rows = []
    for name, fn in variants.items():
        times_us, peak_mb = time_graph(fn, iters), peak_memory_mb(fn)
        err = score_error(ref_scores, fn()) if name != "lmhead_topk" else {}
        for rep, t in enumerate(times_us):
            rows.append(dict(variant=name, batch=batch, rep=rep, time_us=t,
                             precision=precision,
                             peak_mb=peak_mb, **err))
        print(f"B={batch:4d} {name:13s} median {statistics.median(times_us):9.1f}us "
              f"peak {peak_mb:8.1f}MB {err}", flush=True)
    return rows


# --- PR code (vllm/model_executor/models/qwen3_dspine.py @ c1ef7580c6) ---


class Injection(nn.Module):
    """`AdjacentCausalInjection`, with nn.Linear in place of ReplicatedLinear."""

    def __init__(self) -> None:
        super().__init__()
        self.transfer_dim = T
        self.eps = EPS
        lin = lambda i, o: nn.Linear(i, o, bias=False, dtype=DTYPE)  # noqa: E731
        self.readouts = nn.ModuleList(lin(H, T) for _ in range(L))
        self.writes = nn.ModuleList(lin(M, H) for _ in range(L))
        self.message = lin(T, M)
        self.receiver_gate = lin(T, M)
        self.predecessor_gate = lin(T, M)
        self.gate_bias = nn.Parameter(torch.randn(M, dtype=DTYPE) * 0.1)

    @staticmethod
    def rms(states: torch.Tensor) -> torch.Tensor:
        return (
            states.float()
            .square()
            .mean(-1, keepdim=True)
            .clamp_min(torch.finfo(torch.float32).tiny)
            .sqrt()
        )

    def project(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = hidden_states.float() / (self.rms(hidden_states) + self.eps)
        projected = self.readouts[layer_idx](normalized.to(hidden_states.dtype)).float()
        features = (
            math.sqrt(self.transfer_dim)
            * projected
            / (projected.norm(dim=-1, keepdim=True) + self.eps)
        )
        return features.to(hidden_states.dtype)

    def write(self, layer_idx, receiver_states, receiver_features,
              predecessor_features, *, refine: bool = False):
        def linear(layer, features):
            if refine:
                return F.linear(features, layer.weight.float())
            return layer(features).float()

        gate = torch.sigmoid(
            self.receiver_gate(receiver_features).float()
            + linear(self.predecessor_gate, predecessor_features)
            + self.gate_bias.float()
        )
        message = linear(self.message, predecessor_features)
        gated_message = gate * message
        if not refine:
            gated_message = gated_message.to(receiver_states.dtype)
        update = linear(self.writes[layer_idx], gated_message)
        result = receiver_states.float() + self.rms(receiver_states) * update
        return result if refine else result.to(receiver_states.dtype)


def score_ref(w: "Weights", candidate_ids, final_pre_injection, anchor_token_ids):
    injection, final_layer_idx = w.injection, L - 1
    top_k = candidate_ids.shape[-1]
    predecessor_ids = torch.cat(
        (anchor_token_ids[:, None, None].expand(-1, 1, top_k), candidate_ids[:, :-1]),
        dim=1,
    )
    receiver_features = injection.project(final_layer_idx, final_pre_injection)
    receiver_states = final_pre_injection[:, :, None, :]
    local_ids = candidate_ids.long()
    successor_weights = w.lm_head[local_ids].float()
    exact_features = w.codebook[predecessor_ids].float() * math.sqrt(
        injection.transfer_dim
    )
    refined = injection.write(
        final_layer_idx, receiver_states, receiver_features[:, :, None, :],
        exact_features, refine=True,
    )
    normalized = (
        refined
        / (refined.square().mean(-1, keepdim=True) + NORM_EPS).sqrt()
        * w.norm_weight.float()
    )
    return torch.einsum("bnkh,bnjh->bnkj", normalized, successor_weights)


score_ref_compiled = torch.compile(score_ref, dynamic=False)


# --- Low-rank rewrite ---


@dataclass
class Tables:
    pred_gate: torch.Tensor  # [V, M] predecessor_gate(codebook[v] * sqrt(T))
    message: torch.Tensor  # [V, M] message(codebook[v] * sqrt(T))
    successor: torch.Tensor  # [V, M] (lm_head[v] * g) @ W, W = writes[-1].weight
    gram: torch.Tensor  # [M, M] W^T W
    write_t: torch.Tensor  # [H, M] W in fp32
    write_bf16: torch.Tensor  # [H, M] W as stored


def make_tables(w: "Weights") -> Tables:
    inj = w.injection
    write_w = inj.writes[L - 1].weight.float()
    pg, msg, succ = (torch.empty(V, M, device="cuda") for _ in range(3))
    for lo in range(0, V, 16384):
        hi = min(lo + 16384, V)
        x = w.codebook[lo:hi].float() * math.sqrt(T)
        pg[lo:hi] = F.linear(x, inj.predecessor_gate.weight.float())
        msg[lo:hi] = F.linear(x, inj.message.weight.float())
        succ[lo:hi] = (w.lm_head[lo:hi].float() * w.norm_weight.float()) @ write_w
    return Tables(pg, msg, succ, write_w.T @ write_w, write_w,
                  inj.writes[L - 1].weight.contiguous())


def score_lowrank(w: "Weights", t: Tables, candidate_ids, final_pre_injection,
                  anchor_token_ids):
    """score[k, j] = (r.(g*e_j) + s * u_k.t_j) / sqrt(mean(|r + s W u_k|^2) + eps)."""
    inj = w.injection
    predecessor_ids = torch.cat(
        (anchor_token_ids[:, None, None].expand(-1, 1, K), candidate_ids[:, :-1]), dim=1
    )
    receiver_features = inj.project(L - 1, final_pre_injection)
    r = final_pre_injection.float()  # [B, S, H]
    s = inj.rms(final_pre_injection)  # [B, S, 1]
    gate_in = inj.receiver_gate(receiver_features).float() + inj.gate_bias.float()
    gate = torch.sigmoid(gate_in[:, :, None, :] + t.pred_gate[predecessor_ids])
    u = gate * t.message[predecessor_ids]  # [B, S, K, M]
    unary = torch.einsum(
        "bnh,bnjh->bnj", r * w.norm_weight.float(), w.lm_head[candidate_ids].float()
    )
    cross = torch.einsum("bnkm,bnjm->bnkj", u, t.successor[candidate_ids])
    q = r @ t.write_t  # [B, S, M]
    quad = ((u @ t.gram) * u).sum(-1)  # [B, S, K]
    sq = r.square().sum(-1, keepdim=True) + 2 * s * (u * q[:, :, None]).sum(-1) + (
        s.square() * quad
    )
    inv = (sq / H + NORM_EPS).rsqrt()  # [B, S, K]
    return (unary[:, :, None, :] + s[..., None] * cross) * inv[..., None]


# --- Fused Triton kernel for the low-rank scores ---


def score_fused(w: "Weights", t: Tables, candidate_ids, final_pre_injection,
                anchor_token_ids):
    """`score_lowrank` with the per-(request, position) work in one kernel."""
    inj = w.injection
    batch = candidate_ids.shape[0]
    predecessor_ids = torch.cat(
        (anchor_token_ids[:, None, None].expand(-1, 1, K), candidate_ids[:, :-1]), dim=1
    )
    receiver_features = inj.project(L - 1, final_pre_injection)
    gate_in = inj.receiver_gate(receiver_features).float() + inj.gate_bias.float()
    r = final_pre_injection.reshape(-1, H)
    q = torch.mm(r, t.write_bf16, out_dtype=torch.float32)  # exact products
    out = torch.empty(batch * S, K, K, device=r.device)
    _fused_score_kernel[(batch * S,)](
        r, w.norm_weight, gate_in.reshape(-1, M), q,
        predecessor_ids.reshape(-1, K), candidate_ids.reshape(-1, K),
        t.pred_gate, t.message, t.successor, w.lm_head, t.gram, out,
        H, NORM_EPS, w.lm_head.stride(0),
        K=K, M=M, BLOCK_H=512, BLOCK_M=64, num_warps=8,
    )
    return out.view(batch, S, K, K)


@triton.jit
def _fused_score_kernel(
    r_ptr, g_ptr, gate_ptr, q_ptr, pred_ptr, cand_ptr,
    pg_ptr, msg_ptr, succ_ptr, lm_ptr, gram_ptr, out_ptr,
    H, eps, stride_lm,
    K: tl.constexpr, M: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    ks = tl.arange(0, K)
    pred = tl.load(pred_ptr + pid * K + ks)
    cand = tl.load(cand_ptr + pid * K + ks)
    # r.(g*e_j), |r|^2 over the hidden size.
    hs = tl.arange(0, BLOCK_H)
    unary = tl.zeros((K,), tl.float32)
    rr = tl.zeros((BLOCK_H,), tl.float32)
    for h in range(0, H, BLOCK_H):
        r = tl.load(r_ptr + pid * H + h + hs).to(tl.float32)
        rr += r * r
        rg = r * tl.load(g_ptr + h + hs).to(tl.float32)
        e = tl.load(lm_ptr + cand[:, None] * stride_lm + h + hs[None, :]).to(tl.float32)
        unary += tl.sum(e * rg[None, :], 1)
    r_sq = tl.sum(rr, 0)
    s = tl.sqrt(tl.maximum(r_sq / H, 1.1754943508222875e-38))
    # Message-space terms: u_k.t_j, q.u_k and u_k^T G u_k, in BLOCK_M chunks.
    cross = tl.zeros((K, K), tl.float32)
    uq = tl.zeros((K,), tl.float32)
    quad = tl.zeros((K,), tl.float32)
    for c in range(0, M, BLOCK_M):
        u_c = _u_chunk(gate_ptr, pg_ptr, msg_ptr, pid, pred, c, M, BLOCK_M)
        cs = c + tl.arange(0, BLOCK_M)
        succ = tl.load(succ_ptr + cand[:, None] * M + cs[None, :])
        cross += tl.dot(u_c, tl.trans(succ), input_precision="tf32x3")
        uq += tl.sum(u_c * tl.load(q_ptr + pid * M + cs)[None, :], 1)
        ug = tl.zeros((K, BLOCK_M), tl.float32)
        for rb in range(0, M, BLOCK_M):
            u_r = _u_chunk(gate_ptr, pg_ptr, msg_ptr, pid, pred, rb, M, BLOCK_M)
            rbs = rb + tl.arange(0, BLOCK_M)
            gram = tl.load(gram_ptr + rbs[:, None] * M + cs[None, :])
            ug += tl.dot(u_r, gram, input_precision="tf32x3")
        quad += tl.sum(ug * u_c, 1)
    sq = r_sq + 2 * s * uq + s * s * quad
    inv = 1 / tl.sqrt(sq / H + eps)
    scores = (unary[None, :] + s * cross) * inv[:, None]
    tl.store(out_ptr + pid * K * K + ks[:, None] * K + ks[None, :], scores)


@triton.jit
def _u_chunk(gate_ptr, pg_ptr, msg_ptr, pid, pred, c, M: tl.constexpr,
             BLOCK_M: tl.constexpr):
    cs = c + tl.arange(0, BLOCK_M)
    gate_in = tl.load(gate_ptr + pid * M + cs)
    pg = tl.load(pg_ptr + pred[:, None] * M + cs[None, :])
    msg = tl.load(msg_ptr + pred[:, None] * M + cs[None, :])
    return tl.sigmoid(gate_in[None, :] + pg) * msg


def score_fused2(w: "Weights", t: Tables, candidate_ids, final_pre_injection,
                 anchor_token_ids):
    """Per-position kernel for everything but u^T G u, then a GEMM-style kernel
    for u^T G u that also normalizes the scores."""
    inj = w.injection
    batch = candidate_ids.shape[0]
    positions = batch * S
    predecessor_ids = torch.cat(
        (anchor_token_ids[:, None, None].expand(-1, 1, K), candidate_ids[:, :-1]), dim=1
    )
    receiver_features = inj.project(L - 1, final_pre_injection)
    gate_in = inj.receiver_gate(receiver_features).float() + inj.gate_bias.float()
    r = final_pre_injection.reshape(-1, H)
    q = torch.mm(r, t.write_bf16, out_dtype=torch.float32)
    u = torch.empty(positions * K, M, device=r.device)
    numer = torch.empty(positions, K, K, device=r.device)
    lin = torch.empty(positions, K, device=r.device)
    s = torch.empty(positions, device=r.device)
    _position_kernel[(positions,)](
        r, w.norm_weight, gate_in.reshape(-1, M), q,
        predecessor_ids.reshape(-1, K), candidate_ids.reshape(-1, K),
        t.pred_gate, t.message, t.successor, w.lm_head, u, numer, lin, s,
        H, w.lm_head.stride(0), K=K, M=M, BLOCK_H=512, BLOCK_M=64, num_warps=4,
    )
    if QUAD_TF32:
        # u^T G u only scales row k of the scores, so TF32 cannot change the
        # argmax over successors, and its ~1e-3 relative error barely changes
        # the softmax.
        prev = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("high")
        quad = (torch.mm(u, t.gram) * u).sum(-1).view(positions, K)
        torch.set_float32_matmul_precision(prev)
        inv = ((lin + s[:, None].square() * quad) / H + NORM_EPS).rsqrt()
        return (numer * inv[..., None]).view(batch, S, K, K)
    out = torch.empty(positions, K, K, device=r.device)
    block_r = 32
    _quad_norm_kernel[(triton.cdiv(positions * K, block_r),)](
        u, t.gram, numer, lin, s, out, positions * K, H, NORM_EPS,
        K=K, M=M, BLOCK_R=block_r, BLOCK_N=64, BLOCK_K=64, num_warps=4,
    )
    return out.view(batch, S, K, K)


QUAD_TF32 = False


def score_fused3(*args):
    """`score_fused2` with u^T G u as a TF32 cuBLAS GEMM."""
    global QUAD_TF32
    QUAD_TF32 = True
    try:
        return score_fused2(*args)
    finally:
        QUAD_TF32 = False


@triton.jit
def _position_kernel(
    r_ptr, g_ptr, gate_ptr, q_ptr, pred_ptr, cand_ptr,
    pg_ptr, msg_ptr, succ_ptr, lm_ptr, u_ptr, numer_ptr, lin_ptr, s_ptr,
    H, stride_lm,
    K: tl.constexpr, M: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    ks = tl.arange(0, K)
    pred = tl.load(pred_ptr + pid * K + ks)
    cand = tl.load(cand_ptr + pid * K + ks)
    hs = tl.arange(0, BLOCK_H)
    unary = tl.zeros((K,), tl.float32)
    rr = tl.zeros((BLOCK_H,), tl.float32)
    for h in range(0, H, BLOCK_H):
        r = tl.load(r_ptr + pid * H + h + hs).to(tl.float32)
        rr += r * r
        rg = r * tl.load(g_ptr + h + hs).to(tl.float32)
        e = tl.load(lm_ptr + cand[:, None] * stride_lm + h + hs[None, :]).to(tl.float32)
        unary += tl.sum(e * rg[None, :], 1)
    r_sq = tl.sum(rr, 0)
    s = tl.sqrt(tl.maximum(r_sq / H, 1.1754943508222875e-38))
    cross = tl.zeros((K, K), tl.float32)
    uq = tl.zeros((K,), tl.float32)
    for c in range(0, M, BLOCK_M):
        u_c = _u_chunk(gate_ptr, pg_ptr, msg_ptr, pid, pred, c, M, BLOCK_M)
        cs = c + tl.arange(0, BLOCK_M)
        tl.store(u_ptr + (pid * K + ks[:, None]) * M + cs[None, :], u_c)
        succ = tl.load(succ_ptr + cand[:, None] * M + cs[None, :])
        cross += tl.dot(u_c, tl.trans(succ), input_precision="tf32x3")
        uq += tl.sum(u_c * tl.load(q_ptr + pid * M + cs)[None, :], 1)
    tl.store(numer_ptr + pid * K * K + ks[:, None] * K + ks[None, :],
             unary[None, :] + s * cross)
    tl.store(lin_ptr + pid * K + ks, r_sq + 2 * s * uq)
    tl.store(s_ptr + pid, s)


@triton.jit
def _quad_norm_kernel(
    u_ptr, gram_ptr, numer_ptr, lin_ptr, s_ptr, out_ptr, R, H, eps,
    K: tl.constexpr, M: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
    valid = rows < R
    quad = tl.zeros((BLOCK_R,), tl.float32)
    for nb in range(0, M, BLOCK_N):
        ns = nb + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_R, BLOCK_N), tl.float32)
        for kb in range(0, M, BLOCK_K):
            kk = kb + tl.arange(0, BLOCK_K)
            a = tl.load(u_ptr + rows[:, None] * M + kk[None, :], mask=valid[:, None])
            b = tl.load(gram_ptr + kk[:, None] * M + ns[None, :])
            acc += tl.dot(a, b, input_precision="tf32x3")
        u_n = tl.load(u_ptr + rows[:, None] * M + ns[None, :], mask=valid[:, None])
        quad += tl.sum(acc * u_n, 1)
    s = tl.load(s_ptr + rows // K, mask=valid)
    sq = tl.load(lin_ptr + rows, mask=valid) + s * s * quad
    inv = 1 / tl.sqrt(sq / H + eps)
    js = tl.arange(0, K)
    numer = tl.load(numer_ptr + rows[:, None] * K + js[None, :], mask=valid[:, None])
    tl.store(out_ptr + rows[:, None] * K + js[None, :], numer * inv[:, None],
             mask=valid[:, None])


# --- Yardstick ---


def lmhead_topk(w: "Weights", batch: int) -> torch.Tensor:
    logits = w.draft_hidden[: batch * S] @ w.lm_head.T
    return torch.topk(logits, K, dim=-1).indices


# --- Setup and measurement ---


@dataclass
class Weights:
    injection: Injection
    lm_head: torch.Tensor  # [V, H]
    codebook: torch.Tensor  # [V, T]
    norm_weight: torch.Tensor  # [H]
    draft_hidden: torch.Tensor  # [max_B * S, H]


def make_weights() -> Weights:
    injection = Injection().cuda().requires_grad_(False)
    lm_head = torch.randn(V, H, device="cuda", dtype=DTYPE) * 0.02
    codebook = F.normalize(torch.randn(V, T, device="cuda"), dim=-1).to(DTYPE)
    norm_weight = (1 + 0.1 * torch.randn(H, device="cuda")).to(DTYPE)
    draft_hidden = torch.randn(256 * S, H, device="cuda", dtype=DTYPE)
    return Weights(injection, lm_head, codebook, norm_weight, draft_hidden)


def make_inputs(batch: int):
    candidate_ids = torch.randint(0, V, (batch, S, K), device="cuda")
    final_pre_injection = torch.randn(batch, S, H, device="cuda", dtype=DTYPE) * 2
    anchor_token_ids = torch.randint(0, V, (batch,), device="cuda")
    return candidate_ids, final_pre_injection, anchor_token_ids


def time_graph(fn: Callable[[], torch.Tensor], iters: int, reps: int = 5) -> list[float]:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        graph.replay()
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / iters)
    del graph
    return times


def peak_memory_mb(fn: Callable[[], torch.Tensor]) -> float:
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del out
    return peak / 2**20


def score_error(ref: torch.Tensor, got: torch.Tensor) -> dict:
    diff = (got.float() - ref).abs()
    scale = ref.std().item()
    argmax_match = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
    return dict(max_abs_err=diff.max().item(), rel_err=diff.max().item() / scale,
                argmax_match=argmax_match)


def profile_variant(w: Weights, tables: Tables, name: str, batch: int) -> None:
    inputs = make_inputs(batch)
    from torch.profiler import ProfilerActivity, profile

    fn = {"ref": score_ref, "lowrank": lambda *a: score_lowrank(w, tables, *a[1:]),
          "fused": lambda *a: score_fused(w, tables, *a[1:]),
          "fused2": lambda *a: score_fused2(w, tables, *a[1:]),
          "fused3": lambda *a: score_fused3(w, tables, *a[1:])}[name]
    for _ in range(2):
        fn(w, *inputs)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(w, *inputs)
        torch.cuda.synchronize()
    print(f"\nProfile of {name} at B={batch}:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


def write_csv(path: str, rows: list[dict]) -> None:
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
