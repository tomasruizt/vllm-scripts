"""Probe the renorm pipeline on captured Qwen logits, and p=1 rows under both accumulation modes."""

import argparse
import itertools
import json
from pathlib import Path

import torch

from renorm_impls import renorm_probs
from validate_renorm import load_benchmark, total_variation, triton_hint

P1_VOCABS = [65536, 131072, 151936, 248320, 262144]
DISTRIBUTIONS = ["zipf", "normal"]
BATCH = 64


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    module = load_benchmark()
    rows = [*probe_p1(module), *probe_captures(module, args.captures)]
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))


def probe_p1(module) -> list[dict]:
    return [p1_row(module, *case) for case in itertools.product(DISTRIBUTIONS, P1_VOCABS, (True, False))]


def p1_row(module, distribution: str, vocab: int, is_deterministic: bool) -> dict:
    torch.manual_seed(0)
    logits = module.create_logits(BATCH, vocab, distribution=distribution)
    p = torch.ones(BATCH, device="cuda", dtype=torch.float32)
    kept = renorm_probs(logits, None, p, is_deterministic) != 0
    lost = logits.double().softmax(-1).masked_fill(kept, 0.0).sum(-1)
    return dict(kind="p1", distribution=distribution, vocab=vocab, is_deterministic=is_deterministic,
                max_lost_mass=lost.max().item(), mean_lost_mass=lost.mean().item(),
                max_dropped_tokens=(~kept).sum(-1).max().item())


def probe_captures(module, path: Path) -> list[dict]:
    return [capture_row(module, file) for file in sorted(path.rglob("logits-*.pt"))]


def capture_row(module, file: Path) -> dict:
    item = torch.load(file, map_location="cuda", weights_only=True)
    logits, k, p = item["logits"].float(), item["k"], item["p"]
    triton = module.apply_top_k_top_p_triton(logits.clone(), k, p, **triton_hint(k, p, logits.shape[-1])) != -float("inf")
    reference = module.apply_top_k_top_p_pytorch(logits.clone(), k, p) != -float("inf")
    kept = renorm_probs(logits, k, p, True) != 0
    bounds = dict(top_k_min=int(k.min()), top_k_max=int(k.max()))
    fast = module.apply_top_k_top_p_flashinfer_fastpath(logits.clone(), k, p, **bounds) != -float("inf")
    probs = logits.double().softmax(-1)
    return dict(kind="capture", file=file.name, rows=logits.shape[0], vocab=logits.shape[1],
                renorm_vs_triton_rows=(kept != triton).any(-1).sum().item(),
                renorm_vs_reference_rows=(kept != reference).any(-1).sum().item(),
                fastpath_vs_triton_rows=(fast != triton).any(-1).sum().item(),
                renorm_tv_vs_triton=total_variation(probs, kept, triton),
                fastpath_tv_vs_triton=total_variation(probs, fast, triton))


if __name__ == "__main__":
    main()
