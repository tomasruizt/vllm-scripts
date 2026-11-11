"""Optimized FlashInfer fast path, and a timing breakdown of both fast path versions.

The optimized version keeps the fast path's algorithm (FlashInfer top-k selection of
k_max candidates, then top-k/top-p over the candidates) and applies what the CUDA
filter does: mask in place (no new [batch, vocab] tensor), unsorted and
non-deterministic selection (the tie-break alone fixes the selected set), and an exact
order (logit desc, index asc) sorted inside the scatter kernel.

Usage, from the PR tree: python fi_opt.py OUT.jsonl --logits-file ROWS.pt [--batch-sizes ...]
"""

import argparse
import os
from pathlib import Path

import flashinfer
import torch
from fastpath_bench import BATCHES, supported
from fi_fastpath import apply_top_k_top_p_flashinfer
from large_k import load_benchmark

from vllm.triton_utils import tl, triton


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--logits-file", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=BATCHES)
    parser.add_argument("--implementations", nargs="+", default=list(VARIANTS) + ["cuda", "triton"])
    args = parser.parse_args()
    bench = load_benchmark(Path("benchmarks/benchmark_topk_topp.py"))
    real_rows = torch.load(args.logits_file).float().cuda()
    configs = bench.create_benchmark_configs(args.batch_sizes, [131072, 248320])
    configs += bench.create_real_configs(args.batch_sizes, real_rows.shape[1])
    configs = [c for c in configs if supported(c)]
    register(bench)
    bench.run_benchmark(configs, args.implementations, check=True, output=args.output, real_rows=real_rows)


VARIANTS = {
    "fi-fastpath": lambda logits, k, p, bounds: apply_top_k_top_p_flashinfer(logits, k, p, bounds),
    "fi-opt": lambda logits, k, p, bounds: fastpath_opt(logits, k, p, bounds, None),
    "fi-opt-cub": lambda logits, k, p, bounds: fastpath_opt(logits, k, p, bounds, "cub"),
    "fi-opt-clusters": lambda logits, k, p, bounds: fastpath_opt(logits, k, p, bounds, "clusters"),
    "fi-topk-only": lambda logits, k, p, bounds: select(logits, bounds[1], None) and logits,
    "fill-only": lambda logits, k, p, bounds: logits.fill_(float("-inf")),
    "clone-only": lambda logits, k, p, bounds: logits,
}


def register(bench) -> None:
    """The fast paths need the host-side k bounds, computed outside the timing."""
    implementation = bench.implementation

    def with_variants(name: str, config):
        if name not in VARIANTS:
            return implementation(name, config)
        bounds = (int(config.k_values.min()), int(config.k_values.max()))
        return lambda logits, k, p: VARIANTS[name](logits, k, p, bounds)

    bench.IMPLEMENTATIONS.update(VARIANTS)
    bench.implementation = with_variants


def fastpath_opt(logits, k, p, bounds: tuple[int, int], algo: str | None) -> torch.Tensor:
    top_k_min, top_k_max = bounds
    values, indices = select(logits, top_k_max, algo)
    logits.fill_(float("-inf"))
    if p is None and top_k_min == top_k_max:
        return logits.scatter_(1, indices, values)
    _filter_scatter_kernel[(logits.shape[0],)](
        values, indices, k, p, logits, logits.stride(0), k.stride(0),
        p.stride(0) if p is not None else 0, top_k_max, p is not None,
        triton.next_power_of_2(top_k_max), num_warps=4,
    )  # fmt: skip
    return logits


def select(logits: torch.Tensor, top_k_max: int, algo: str | None):
    """Unsorted top-k_max candidates, lowest index first at the boundary."""
    if algo is None:
        os.environ.pop("FLASHINFER_TOPK_ALGO", None)
    else:
        os.environ["FLASHINFER_TOPK_ALGO"] = algo
    return flashinfer.top_k(logits, top_k_max, sorted=False, deterministic=False, tie_break=1)


@triton.jit
def _filter_scatter_kernel(
    values_ptr, indices_ptr, k_ptr, p_ptr, output_ptr, output_row_stride, k_stride,
    p_stride, TOP_K_MAX: tl.constexpr, HAS_P: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    """Sort a row's candidates by (logit desc, index asc), cut at k and p, scatter."""
    row = tl.program_id(0)
    offset = tl.arange(0, BLOCK)
    valid = offset < TOP_K_MAX
    value = tl.load(values_ptr + row * TOP_K_MAX + offset, mask=valid, other=-float("inf"))
    index = tl.load(indices_ptr + row * TOP_K_MAX + offset, mask=valid, other=0).to(tl.int64)
    # Sort key, ascending: order-preserving uint32 of the logit, inverted, with the
    # index in the low bits, so the largest logit comes first and ties by index.
    bits = tl.where(value == 0, 0.0, value).to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    ordered = tl.where(bits >= 0x80000000, 0xFFFFFFFF - bits, bits + 0x80000000)
    # Signed high word: 0x7FFFFFFF - ordered keeps the unsigned order.
    key = tl.where(valid, ((0x7FFFFFFF - ordered) << 32) | index, 0x7FFFFFFFFFFFFFFF)
    key = tl.sort(key, 0)
    index = key & 0xFFFFFFFF
    ordered = 0x7FFFFFFF - (key >> 32)
    bits = tl.where(ordered >= 0x80000000, ordered - 0x80000000, 0xFFFFFFFF - ordered)
    value = bits.to(tl.uint32).to(tl.float32, bitcast=True)
    keep = offset < tl.load(k_ptr + row * k_stride)
    if HAS_P:
        weight = tl.where(keep, tl.exp(value - tl.max(tl.where(keep, value, -float("inf")), 0)), 0.0)
        before = tl.cumsum(weight, 0) - weight
        keep = keep & (before < tl.load(p_ptr + row * p_stride) * tl.sum(weight, 0))
    tl.store(output_ptr + row * output_row_stride + index, value, mask=keep)


if __name__ == "__main__":
    main()
