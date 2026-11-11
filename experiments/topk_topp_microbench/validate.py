"""Differential validation, with exact native compatibility as a separate gate."""

import argparse
import itertools
import json
import os
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import flashinfer
import torch

from benchmark import compact_filter, native, apply_top_k_top_p_pytorch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["small", "large", "captured"], default="small")
    parser.add_argument("--captures", type=Path)
    parser.add_argument("--backends", nargs="+", default=["pytorch", "flashinfer"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    cases = captured_cases(args.captures) if args.phase == "captured" else synthetic_cases(args.phase)
    failures = 0
    count = 0
    with (args.output / "validation.jsonl").open("x") as file:
        for case, source, ks, ps in cases:
            for row in validate_case(case, source, ks, ps, args.backends):
                file.write(json.dumps(row) + "\n")
                failures += not row.get("native_exact", False)
                count += 1
            file.flush()
            if count % 100 == 0:
                print(json.dumps(dict(comparisons=count, native_mismatches=failures)), flush=True)
    summary = dict(comparisons=count, native_mismatches=failures, phase=args.phase,
                   torch=torch.__version__, flashinfer=flashinfer.__version__,
                   gpu=torch.cuda.get_device_name(), gpu_ids=os.environ.get("CUDA_VISIBLE_DEVICES"))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "DONE").write_text("complete\n")
    print(json.dumps(summary), flush=True)
    raise SystemExit(2 if failures else 0)


@torch.inference_mode()
def validate_case(case: dict, source: torch.Tensor, ks: list[int] | None, ps: list[float] | None,
                  backends: Sequence[str] = ("pytorch", "flashinfer")) -> list[dict]:
    k = torch.tensor(ks, device="cuda", dtype=torch.int32) if ks is not None else None
    p = torch.tensor(ps, device="cuda", dtype=torch.float32) if ps is not None else None
    original = native.apply_top_k_top_p_triton(with_layout(source.clone(), case["layout"]), k, p)
    reference = apply_top_k_top_p_pytorch(with_layout(source.clone(), case["layout"]), k, p)
    rows = []
    for backend in backends:
        try:
            actual, route = candidate(with_layout(source.clone(), case["layout"]), ks, k, p, backend)
            row = compare(source, actual, original, reference)
            rows.append(dict(**case, backend=backend, route=route, **row, error=None))
        except Exception as error:
            rows.append(dict(**case, backend=backend, route="error", native_exact=False,
                             error=f"{type(error).__name__}: {error}"))
    return rows


def candidate(source: torch.Tensor, ks: list[int] | None, k: torch.Tensor | None,
              p: torch.Tensor | None, backend: str) -> tuple[torch.Tensor, str]:
    scalar = ks[0] if ks is not None else None
    eligible = scalar is not None and 0 < scalar <= 128 and scalar < source.shape[-1] and all(v == scalar for v in ks)
    if backend.startswith("flashinfer") and source.shape[0] > 1 and source.shape[-1] % 4:
        return native.apply_top_k_top_p_triton(source, k, p), "alignment-fallback"
    if not eligible:
        return native.apply_top_k_top_p_triton(source, k, p), "native-fallback"
    select = torch.topk if backend == "pytorch" else flashinfer.top_k
    if backend == "flashinfer-deterministic":
        select = partial(flashinfer.top_k, deterministic=True, tie_break=1)
    return compact_filter(source, select, p, scalar), "compact"


def compare(source: torch.Tensor, actual: torch.Tensor, original: torch.Tensor, reference: torch.Tensor) -> dict:
    mask = actual > -float("inf")
    finite_rows = torch.isfinite(source).any(-1)
    basic = (actual.shape == source.shape and actual.dtype == source.dtype
             and not bool(actual.isnan().any())
             and bool(torch.equal(actual[mask], source[mask]))
             and bool((mask.any(-1) | ~finite_rows).all()))
    prob = actual.softmax(-1)
    native_prob = original.softmax(-1)
    ref_prob = reference.softmax(-1)
    valid = finite_rows & torch.isfinite(prob).all(-1) & torch.isfinite(native_prob).all(-1) & torch.isfinite(ref_prob).all(-1)
    tv_native = float((prob[valid] - native_prob[valid]).abs().sum(-1).max()) / 2 if bool(valid.any()) else None
    tv_reference = float((prob[valid] - ref_prob[valid]).abs().sum(-1).max()) / 2 if bool(valid.any()) else None
    removed_max = torch.where(mask, -float("inf"), source).amax(-1)
    kept_min = torch.where(mask, source, float("inf")).amin(-1)
    return dict(basic_valid=basic, native_exact=bool(torch.equal(actual, original)),
                reference_exact=bool(torch.equal(actual, reference)), native_tv=tv_native,
                reference_tv=tv_reference, native_ref_tv=max_tv(native_prob, ref_prob, valid),
                ranking_valid=bool(((kept_min >= removed_max) | ~finite_rows).all()),
                kept_min=int(mask.sum(-1).min()), kept_max=int(mask.sum(-1).max()),
                native_kept_min=int((original > -float("inf")).sum(-1).min()),
                native_kept_max=int((original > -float("inf")).sum(-1).max()))


def max_tv(a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor) -> float | None:
    return float((a[valid] - b[valid]).abs().sum(-1).max()) / 2 if bool(valid.any()) else None


def synthetic_cases(phase: str):
    batches = [1, 4, 9] if phase == "small" else [4, 9, 192, 256, 280]
    vocabs = [257, 4096] if phase == "small" else [248320]
    kvals = [1, 2, 20, 32, 128] if phase == "small" else [1, 20, 64]
    pvals = [None, 0.01, 0.1, 0.5, 0.8, 0.95, 1.0] if phase == "small" else [0.1, 0.8, 1.0]
    kinds = ["normal", "bf16", "sparse", "equal", "peaked", "near-equal", "offset"] if phase == "small" else ["normal", "bf16", "sparse"]
    for seed, batch, vocab, kind in itertools.product([0, 42], batches, vocabs, kinds):
        torch.manual_seed(seed)
        source = make_logits(batch, vocab, kind)
        for index, (top_k, top_p) in enumerate(itertools.product(kvals, pvals)):
            layout = ["contiguous", "row-stride", "column-stride"][index % 3]
            case = dict(seed=seed, batch=batch, vocab=vocab, kind=kind, top_k=top_k, top_p=top_p, layout=layout)
            yield case, with_layout(source, layout), [top_k] * batch, None if top_p is None else [top_p] * batch
    for vocab in vocabs:
        source = make_logits(4, vocab, "normal")
        settings = [(None, None), (None, [0.5, 0.8, 0.95, 1]),
                    ([1, 20, vocab, 64], [1, 0.8, 0.95, 0.5]), ([vocab] * 4, [0.8] * 4)]
        for ks, ps in settings:
            case = dict(seed=42, batch=4, vocab=vocab, kind="mixed-parameters", top_k=None, top_p=None, layout="contiguous")
            yield case, source, ks, ps
    source = torch.full((4, 257), -float("inf"), device="cuda")
    yield dict(seed=42, batch=4, vocab=257, kind="all-masked", top_k=20, top_p=0.8, layout="contiguous"), source, [20]*4, [0.8]*4


def make_logits(batch: int, vocab: int, kind: str) -> torch.Tensor:
    source = torch.randn(batch, vocab, device="cuda") / 0.7
    if kind == "bf16":
        return source.bfloat16().float()
    if kind == "equal":
        return torch.ones_like(source)
    if kind == "near-equal":
        return torch.ones_like(source) + source * 1e-6
    if kind == "offset":
        return source + 100000
    if kind == "peaked":
        source[:, 0] = 100
    if kind == "sparse":
        source[:, 13:] = -float("inf")
    return source


def with_layout(source: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "contiguous":
        return source
    extra = 3 if layout == "row-stride" else source.shape[-1]
    backing = source.new_empty((source.shape[0], source.shape[1] + extra))
    view = backing[:, :source.shape[1]] if layout == "row-stride" else backing[:, ::2]
    view.copy_(source)
    return view


def captured_cases(path: Path):
    for file in sorted(path.rglob("logits-*.pt")):
        item = torch.load(file, map_location="cuda", weights_only=True)
        source = item["logits"].float()
        ks = item["k"].cpu().tolist() if item["k"] is not None else None
        ps = item["p"].cpu().tolist() if item["p"] is not None else None
        case = dict(seed=42, batch=source.shape[0], vocab=source.shape[1], kind="captured", top_k=ks[0] if ks else None,
                    top_p=ps[0] if ps else None, layout="contiguous", capture=str(file), original_rows=item["original_rows"])
        yield case, source, ks, ps


if __name__ == "__main__":
    main()
