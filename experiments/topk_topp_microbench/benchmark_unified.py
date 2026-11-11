"""Validate and time the experimental unified filter using vLLM's benchmark."""

import argparse
from contextlib import redirect_stdout
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import torch

from sweep_existing import metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--warp-aggregate", action="store_true")
    args = parser.parse_args()
    warnings.filterwarnings("error", message=".*Falling back to CUDA events.*")
    from unified_topk_topp import UnifiedFilter

    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    prototype = UnifiedFilter(warp_aggregate=args.warp_aggregate)
    print(json.dumps(dict(compiled_s=time.monotonic() - started)), flush=True)
    spec = importlib.util.spec_from_file_location("benchmark_helpers", "benchmark_topk_topp.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.IMPLEMENTATIONS["unified"] = prototype
    info = metadata()
    info["warp_aggregate"] = args.warp_aggregate
    for filename in ["unified_topk_topp.cu", "unified_topk_topp.py", "benchmark_unified.py"]:
        info["sources"][filename] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    (args.output / "metadata.json").write_text(json.dumps(info, indent=2))
    rows = validate(module, prototype, args.smoke_only)
    (args.output / "validation.json").write_text(json.dumps(rows, indent=2))
    failures = sum(not row["valid"] for row in rows)
    print(json.dumps(dict(cases=len(rows), failures=failures)), flush=True)
    if failures:
        raise SystemExit(2)
    if not args.smoke_only:
        started = time.monotonic()
        for repeat in range(3):
            execute(module, args.output, repeat)
        (args.output / "summary.json").write_text(json.dumps(dict(elapsed_s=time.monotonic() - started)))
    (args.output / "DONE").write_text("ok\n")


def validate(module, prototype, smoke_only: bool) -> list[dict]:
    torch.manual_seed(13)
    rows = []
    vocabs = [257] if smoke_only else [257, 4096, 65536, 248320]
    for vocab in vocabs:
        batches = [4] if smoke_only else [1, 4, 128]
        for config in module.create_benchmark_configs(batches, [vocab], top_k=20, top_p=0.8):
            logits = module.create_logits(config.batch_size, vocab, distribution="normal")
            k, p = config.k_values, config.p_values
            expected = module.apply_top_k_top_p_pytorch(logits.clone(), k, p).softmax(-1)
            masked = prototype(logits.clone(), k, p)
            actual = masked.softmax(-1)
            reference_close = torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
            valid = oracle_valid(logits, masked, k, p)
            row = dict(case=config.name, valid=bool(valid),
                       reference_close=bool(reference_close),
                       max_tv=float((actual - expected).abs().sum(-1).max()) / 2)
            rows.append(row)
            print(json.dumps(row), flush=True)
    if not smoke_only:
        rows.extend(validate_edges(prototype))
    return rows


def validate_edges(prototype) -> list[dict]:
    rows = []
    vocab = 65537
    sparse = torch.full((4, vocab), -float("inf"), device="cuda")
    sparse[:, :13] = torch.randn(4, 13, device="cuda")
    boundary = torch.full_like(sparse, -float("inf"))
    boundary[:, :3] = torch.tensor([0.5, 0.25, 0.25], device="cuda").log()
    sources = [("sparse", sparse), ("uniform", torch.zeros_like(sparse)),
               ("all-masked", torch.full_like(sparse, -float("inf"))),
               ("boundary", boundary), ("rounded", torch.randn_like(sparse).bfloat16().float())]
    for name, source in sources:
        for ks in ([20] * 4, [1, 20, 256, vocab], [vocab] * 4):
            k = torch.tensor(ks, device="cuda", dtype=torch.int32)
            for ps in (None, [0.1, 0.5, 0.8, 1.0], [0.75] * 4):
                p = None if ps is None else torch.tensor(ps, device="cuda")
                output = prototype(source.clone(), k, p)
                valid = oracle_valid(source, output, k, p)
                row = dict(case=name, k=ks, p=ps, valid=valid)
                rows.append(row)
                print(json.dumps(row), flush=True)
    # Capture after workspace initialization on the capture stream.
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    source = torch.randn((4, vocab), device="cuda")
    k = torch.tensor([1, 20, 256, vocab], device="cuda", dtype=torch.int32)
    p = torch.tensor([0.1, 0.5, 0.8, 1.0], device="cuda")
    captured = source.clone()
    with torch.cuda.stream(stream):
        prototype(captured, k, p)
    stream.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        prototype(captured, k, p)
    for repeat in range(5):
        k.copy_(k.roll(1))
        p.copy_(p.roll(-1))
        captured.copy_(source)
        graph.replay()
        rows.append(dict(case="graph-replay", repeat=repeat,
                         k=k.tolist(), p=p.tolist(),
                         valid=oracle_valid(source, captured, k, p)))
    return rows


def oracle_valid(source, actual, k, p) -> bool:
    batch, vocab = source.shape
    order = source.argsort(dim=-1, descending=True, stable=True)
    sorted_source = source.gather(-1, order).double()
    kept = actual.isfinite().gather(-1, order)
    count = kept.sum(-1)
    ranks = torch.arange(vocab, device=source.device)
    if not torch.equal(kept, ranks[None, :] < count[:, None]):
        return False
    if not torch.equal(actual[actual.isfinite()], source[actual.isfinite()]):
        return False
    finite = source.isfinite().sum(-1)
    budget = finite if k is None else torch.minimum(k.long(), finite)
    if not bool(((count <= budget) & ((count >= 1) | (finite == 0))).all()):
        return False
    if p is None:
        return torch.equal(count, budget)
    values = sorted_source.masked_fill(ranks[None, :] >= budget[:, None], -float("inf"))
    prefix = values.softmax(-1).cumsum(-1)
    mass = prefix.gather(-1, (count - 1).clamp_min(0)[:, None]).squeeze(-1)
    previous = prefix.gather(-1, (count - 2).clamp_min(0)[:, None]).squeeze(-1)
    previous = torch.where(count <= 1, 0, previous)
    valid = torch.where(p >= 1, count == budget, (mass >= p.double() - 1e-7) & (previous <= p.double() + 1e-7))
    return bool((valid | (finite == 0)).all())


def execute(module, output: Path, repeat: int) -> None:
    order = ["triton", "unified"] if repeat != 1 else ["unified", "triton"]
    command = ["benchmark_topk_topp.py", "--batch-sizes", "4", "64", "512",
               "--vocab-sizes", "65536", "131072", "248320", "--top-k", "20",
               "--top-p", "0.8", "--implementations", *order, "--check",
               "--warmup-iters", "5", "--benchmark-iters", "30", "--output",
               str(output / f"case-{repeat:03d}.jsonl")]
    (output / f"case-{repeat:03d}.command.json").write_text(json.dumps(command, indent=2))
    sys.argv = command
    print(json.dumps(dict(repeat=repeat, order=order)), flush=True)
    with (output / f"case-{repeat:03d}.log").open("x") as log, redirect_stdout(log):
        module.main()


if __name__ == "__main__":
    main()
