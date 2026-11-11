"""Check the FlashInfer renorm pipelines against the sampler's masked-logit semantics."""

import argparse
import importlib.util
import itertools
import json
from pathlib import Path
import sys
from types import ModuleType

import torch

from renorm_impls import apply_fi_mask, apply_fi_renorm, renorm_probs

BATCH = 64
VOCABS = [131072, 248320, 262144]
DISTRIBUTIONS = ["zipf", "normal", "bf16-rounded"]
SCALES = [1.0, 20.0]  # 20.0 stands in for T=0.05, enough to underflow FP32 softmax.
SEEDS = [0, 1, 2]
DISABLED_K = "vocab"
SETUPS = [
    dict(name="topk20", k=[20], p=None),
    dict(name="topk256", k=[256], p=None),
    dict(name="topp0.8", k=None, p=[0.8]),
    dict(name="topp1", k=None, p=[1.0]),
    dict(name="topp0.8_or_1", k=None, p=[0.8, 1.0]),
    dict(name="k20_p0.8", k=[20], p=[0.8]),
    dict(name="k100_p0.95", k=[100], p=[0.95]),
    dict(name="k20_or_100_p1_or_0.8", k=[20, 100], p=[1.0, 0.8]),
    dict(name="k20_or_off_p0.8", k=[20, DISABLED_K], p=[0.8]),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    module = load_benchmark()
    cases = itertools.product(DISTRIBUTIONS, VOCABS, SCALES, SEEDS, SETUPS)
    with args.output.open("x") as out:
        for distribution, vocab, scale, seed, setup in cases:
            for row in check_case(module, distribution, vocab, scale, seed, setup):
                out.write(json.dumps(row) + "\n")


def load_benchmark() -> ModuleType:
    spec = importlib.util.spec_from_file_location("renorm_benchmark", "benchmark_topk_topp.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_case(module: ModuleType, distribution: str, vocab: int, scale: float, seed: int, setup: dict) -> list[dict]:
    torch.manual_seed(seed)
    logits = module.create_logits(BATCH, vocab, distribution=distribution) * scale
    k, p = row_tensors(setup, vocab)
    reference = module.apply_top_k_top_p_pytorch(logits.clone(), k, p) != -float("inf")
    triton = module.apply_top_k_top_p_triton(logits.clone(), k, p, **triton_hint(k, p, vocab)) != -float("inf")
    base = dict(distribution=distribution, vocab=vocab, scale=scale, seed=seed, setup=setup["name"],
                reference_kept_mean=reference.sum(-1).float().mean().item())
    impls = {"triton": lambda: triton, "fi-mask": lambda: apply_fi_mask(logits.clone(), k, p) != -float("inf"),
             "fi-renorm": lambda: apply_fi_renorm(logits.clone(), k, p) != -float("inf")}
    if eligible_fastpath(k, vocab):
        impls["flashinfer-fastpath"] = lambda: fastpath(module, logits, k, p)
    rows = [metrics(base, name, run(), logits, k, p, reference, triton) for name, run in impls.items()]
    rows.append(determinism(base, logits, k, p))
    return rows


def row_tensors(setup: dict, vocab: int) -> tuple:
    def rows(values: list | None, dtype):
        if values is None:
            return None
        resolved = [vocab if value == DISABLED_K else value for value in values]
        return torch.tensor([resolved[i % len(resolved)] for i in range(BATCH)], device="cuda", dtype=dtype)

    return rows(setup["k"], torch.int32), rows(setup["p"], torch.float32)


def triton_hint(k: torch.Tensor | None, p: torch.Tensor | None, vocab: int) -> dict:
    if p is None:
        return dict(has_topp_only_rows=False)
    only_p = p < 1 if k is None else (p < 1) & (k >= vocab)
    return dict(has_topp_only_rows=bool(only_p.any().item()))


def eligible_fastpath(k: torch.Tensor | None, vocab: int) -> bool:
    return k is not None and int(k.max()) <= 256 and vocab >= 65536


def fastpath(module: ModuleType, logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor | None) -> torch.Tensor:
    bounds = dict(top_k_min=int(k.min()), top_k_max=int(k.max()))
    return module.apply_top_k_top_p_flashinfer_fastpath(logits.clone(), k, p, **bounds) != -float("inf")


def metrics(base: dict, name: str, kept: torch.Tensor, logits: torch.Tensor, k, p, reference, triton) -> dict:
    probs = logits.double().softmax(dim=-1)
    kept_count = kept.sum(-1)
    row = dict(base, impl=name, kept_mean=kept_count.float().mean().item(),
               reference_mismatch_rows=(kept != reference).any(-1).sum().item(),
               triton_mismatch_rows=(kept != triton).any(-1).sum().item(),
               max_tv=total_variation(probs, kept, reference),
               max_tv_triton=total_variation(probs, kept, triton))
    if k is not None:
        row["rows_over_k"] = (kept_count > k).sum().item()
        row["rows_under_k"] = (kept_count < torch.minimum(k, torch.full_like(k, logits.shape[-1]))).sum().item() if p is None else None
    if p is not None and (p >= 1).any():
        unfiltered = p >= 1 if k is None else (p >= 1) & (k >= logits.shape[-1])
        row.update(disabled_row_stats(probs, kept, unfiltered))
    return row


def total_variation(probs: torch.Tensor, kept: torch.Tensor, other: torch.Tensor) -> float:
    left = probs.masked_fill(~kept, 0.0)
    right = probs.masked_fill(~other, 0.0)
    left = left / left.sum(-1, keepdim=True)
    right = right / right.sum(-1, keepdim=True)
    return ((left - right).abs().sum(-1).max() / 2).item()


def disabled_row_stats(probs: torch.Tensor, kept: torch.Tensor, unfiltered: torch.Tensor) -> dict:
    """Rows with no active filter must keep the whole vocabulary."""
    if not unfiltered.any():
        return {}
    lost = probs.masked_fill(kept, 0.0).sum(-1)[unfiltered]
    dropped = (~kept).sum(-1)[unfiltered]
    return dict(disabled_rows=int(unfiltered.sum()), disabled_max_lost_mass=lost.max().item(),
                disabled_max_dropped_tokens=dropped.max().item())


def determinism(base: dict, logits: torch.Tensor, k, p, repeats: int = 20) -> dict:
    """Bitwise repeatability of the renorm output for identical input."""
    row = dict(base, impl="determinism")
    for flag in (True, False):
        first = renorm_probs(logits, k, p, is_deterministic=flag)
        same = [torch.equal(first, renorm_probs(logits, k, p, is_deterministic=flag)) for _ in range(repeats)]
        row[f"bitwise_stable_is_deterministic_{flag}"] = all(same)
    return row


if __name__ == "__main__":
    main()
