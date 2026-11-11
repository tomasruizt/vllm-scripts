"""Dense top-k/top-p masking + gumbel_sample (or rejection_sample) vs. the compact
kept-token output (top_k_top_p_kept_tokens_triton, then gumbel_sample or
rejection_sample with kept=...) on real Qwen verify rows.

Usage, from the vllm tree: python kept_tokens_bench.py OUT.jsonl --logits-file ROWS.pt
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.v1.sample.ops.topk_topp_triton import (
    apply_top_k_top_p_triton,
    top_k_top_p_kept_tokens_triton,
)
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

ROWS = [8, 16, 64, 128, 256, 1024, 2048]
TOP_K, TOP_P = 20, 0.95
NUM_SPEC = 7


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--logits-file", required=True)
    parser.add_argument("--passes", type=int, default=3)
    args = parser.parse_args()
    real_rows = torch.load(args.logits_file).float().cuda()
    for pass_idx in range(args.passes):
        for rows in ROWS:
            write(args.output, bench_rows(real_rows, rows) | dict(pass_idx=pass_idx))
            for has_draft_logits in (False, True):
                record = bench_verify(real_rows, rows, has_draft_logits)
                write(args.output, record | dict(pass_idx=pass_idx))


def write(output: Path, record: dict) -> None:
    with output.open("a") as file:
        file.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def bench_rows(real_rows: torch.Tensor, rows: int) -> dict:
    logits = tile_rows(real_rows, rows)
    k = torch.full((rows,), TOP_K, dtype=torch.int32, device="cuda")
    p = torch.full((rows,), TOP_P, dtype=torch.float32, device="cuda")
    sample_args = sampling_args(rows)
    arms = make_arms(logits, k, p, sample_args)
    dense, kept = arms["mask+gumbel"](), arms["kept+gumbel"]()
    times = {name: time_us(fn) for name, fn in arms.items()}
    clone = times.pop("clone")
    record = {name: round(t - clone, 1) for name, t in times.items()}
    vocab = logits.shape[1]
    sol_us = 2 * rows * vocab * 4 / 8e12 * 1e6
    return record | dict(
        kind="sample",
        rows=rows,
        vocab=vocab,
        clone_us=round(clone, 1),
        sol_us=round(sol_us, 1),
        same_tokens=bool(torch.equal(dense, kept)),
    )


def make_arms(logits, k, p, sample_args) -> dict[str, Callable[[], torch.Tensor]]:
    def mask():
        return apply_top_k_top_p_triton(logits.clone(), k, p)

    def kept():
        return top_k_top_p_kept_tokens_triton(logits.clone(), k, p, TOP_K)

    def mask_gumbel():
        return gumbel_sample(mask(), *sample_args, apply_temperature=False, is_drafting=False)

    def kept_gumbel():
        return gumbel_sample(
            logits, *sample_args, apply_temperature=False, is_drafting=False, kept=kept()
        )

    return {
        "clone": logits.clone,
        "mask": mask,
        "kept": kept,
        "mask+gumbel": mask_gumbel,
        "kept+gumbel": kept_gumbel,
    }


def bench_verify(real_rows: torch.Tensor, rows: int, has_draft_logits: bool) -> dict:
    """One verify step of rows // (NUM_SPEC + 1) requests: masking + rejection_sample
    vs. kept tokens + rejection_sample(kept=...)."""
    logits = tile_rows(real_rows, rows)
    k = torch.full((rows,), TOP_K, dtype=torch.int32, device="cuda")
    p = torch.full((rows,), TOP_P, dtype=torch.float32, device="cuda")
    inputs = verify_inputs(logits, k, p, has_draft_logits)
    vocab = logits.shape[1]

    def dense():
        masked = apply_top_k_top_p_triton(logits.clone(), k, p)
        return rejection_sample(masked, **inputs, skip_masked_vocab_blocks=True)

    def kept():
        kept_tokens = top_k_top_p_kept_tokens_triton(logits.clone(), k, p, TOP_K)
        return rejection_sample(logits, **inputs, kept=kept_tokens)

    (ref, ref_num), (out, out_num) = dense(), kept()
    valid = torch.arange(NUM_SPEC + 1, device="cuda") < ref_num[:, None]
    same = torch.equal(ref_num, out_num) and torch.equal(ref[valid], out[valid])
    clone = time_us(logits.clone)
    return dict(
        kind="verify",
        has_draft_logits=has_draft_logits,
        rows=rows,
        vocab=vocab,
        dense=round(time_us(dense) - clone, 1),
        kept=round(time_us(kept) - clone, 1),
        mean_accepted=round(float(ref_num.float().mean()) - 1, 2),
        same_tokens=bool(same),
    )


def verify_inputs(logits, k, p, has_draft_logits: bool) -> dict:
    """Drafts sampled from the masked target, so most are accepted."""
    rows, vocab = logits.shape
    num_reqs = rows // (NUM_SPEC + 1)
    masked = apply_top_k_top_p_triton(logits.clone(), k, p)
    gen = torch.Generator("cuda").manual_seed(0)
    draft_sampled = torch.zeros(rows, dtype=torch.int64, device="cuda")
    draft_sampled[1:] = torch.multinomial(masked[:-1].softmax(-1), 1, generator=gen).view(-1)
    draft_sampled[:: NUM_SPEC + 1] = 0
    draft_logits_cache = None
    if has_draft_logits:
        draft_rows = logits.view(num_reqs, NUM_SPEC + 1, vocab)[:, :NUM_SPEC]
        values = (
            draft_rows + torch.randn(draft_rows.shape, device="cuda", generator=gen)
        ).bfloat16()
        draft_logits_cache = DraftLogitsCache(1, 1, 1, dtype=values.dtype, fill=0.0, device="cuda")
        draft_logits_cache.values = values
        draft_logits_cache.row_mapping = None
    arange = torch.arange(num_reqs, dtype=torch.int32, device="cuda")
    return dict(
        draft_logits_cache=draft_logits_cache,
        draft_sampled=draft_sampled,
        cu_num_logits=torch.arange(num_reqs + 1, dtype=torch.int32, device="cuda") * (NUM_SPEC + 1),
        pos=torch.arange(rows, dtype=torch.int32, device="cuda"),
        idx_mapping=arange,
        expanded_idx_mapping=arange.repeat_interleave(NUM_SPEC + 1),
        expanded_local_pos=torch.arange(NUM_SPEC + 1, dtype=torch.int32, device="cuda").repeat(
            num_reqs
        ),
        temperature=torch.ones(num_reqs, dtype=torch.float32, device="cuda"),
        seed=arange.long() * 7919,
        num_speculative_steps=NUM_SPEC,
    )


def sampling_args(rows: int) -> tuple[torch.Tensor, ...]:
    idx_mapping = torch.arange(rows, dtype=torch.int32, device="cuda")
    temperature = torch.ones(rows, dtype=torch.float32, device="cuda")
    seed = torch.arange(rows, dtype=torch.int64, device="cuda") * 7919
    pos = torch.arange(rows, dtype=torch.int64, device="cuda")
    return idx_mapping, temperature, seed, pos


def tile_rows(real_rows: torch.Tensor, rows: int) -> torch.Tensor:
    reps = -(-rows // real_rows.shape[0])
    return real_rows.repeat(reps, 1)[:rows].contiguous()


def time_us(fn: Callable[[], object]) -> float:
    times = bench_gpu_time_with_cupti(
        fn, dry_run_iters=5, repeat_iters=30, use_cuda_graph=False, cold_l2_cache=True
    )
    return sorted(times)[len(times) // 2] * 1e3


if __name__ == "__main__":
    main()
