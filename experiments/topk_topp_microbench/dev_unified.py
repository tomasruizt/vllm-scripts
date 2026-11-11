"""Dev loop: per-row path statistics and kernel-only timings vs Triton."""

import argparse
import importlib
import json
from collections import Counter
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from benchmark_unified import oracle_valid, validate
from benchmark_unified2 import load_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-module", default="unified_topk_topp_v8")
    parser.add_argument("--batches", default="4,64,512")
    parser.add_argument("--vocabs", default="65536")
    parser.add_argument("--distribution", default="zipf")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--force-splits", type=int, default=0)
    parser.add_argument("--flags", type=int, default=0, help="kernel ablation flags")
    parser.add_argument("--top-k", type=int, help="default: benchmark defaults (50/100)")
    parser.add_argument("--top-p", type=float, help="default: benchmark defaults (0.9/0.5)")
    parser.add_argument("--scenarios", help="comma-separated subset")
    parser.add_argument("--captures", help="directory of captured logits (vocab from file)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    props = torch.cuda.get_device_properties(0)
    print(json.dumps(dict(sms=props.multi_processor_count, l2_mb=props.L2_cache_size / 2**20)))
    kernel = importlib.import_module(args.kernel_module)
    options = dict(force_splits=args.force_splits, flags=args.flags)
    prototype = kernel.UnifiedFilter(**{k: v for k, v in options.items() if v})
    if hasattr(prototype.module, "info"):
        print(json.dumps(dict(info=prototype.module.info(0))), flush=True)
    module = load_benchmark()
    if args.validate:
        rows = validate(module, prototype, smoke_only=False)
        failures = [row for row in rows if not row["valid"]]
        print(json.dumps(dict(cases=len(rows), failures=len(failures))), flush=True)
        if failures:
            raise SystemExit(2)
    impls = dict(triton=module.IMPLEMENTATIONS["triton"], unified=prototype)
    batches = [int(x) for x in args.batches.split(",")]
    vocabs = [int(x) for x in args.vocabs.split(",")]
    with (args.output / "dev.jsonl").open("w") as out:
        configs = module.create_benchmark_configs(batches, vocabs, top_k=args.top_k,
                                                  top_p=args.top_p)
        wanted = args.scenarios.split(",") if args.scenarios else None
        for config in configs:
            if wanted and config.name.split("_b")[0] not in wanted:
                continue
            row = run_case(module, impls, prototype, config, args)
            out.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


def run_case(module, impls, prototype, config, args) -> dict:
    source = make_source(module, config, args)
    k, p = config.k_values, config.p_values
    debug = torch.zeros(config.batch_size, 4, dtype=torch.int32, device="cuda")
    output = prototype(source.clone(), k, p, debug=debug)
    paths = Counter(debug[:, 0].tolist())
    row = dict(case=config.name, dist=args.distribution, paths={str(a): b for a, b in paths.items()},
               max_candidates=int(debug[:, 1].max()), restarts=int(debug[:, 3].sum()))
    if args.captures:
        row["valid"] = bool(oracle_valid(source, output, k, p))
    kwargs = module.prepare_kwargs("triton", k, p, config.vocab_size)
    for name, fn in impls.items():
        extra = kwargs if name == "triton" else {}
        row[f"{name}_us"] = kernel_time(lambda x: fn(x, k, p, **extra), source, args.iters)
    return row


def make_source(module, config, args) -> torch.Tensor:
    """Synthetic logits, or captured serving rows tiled to the batch size."""
    if not args.captures:
        return module.create_logits(config.batch_size, config.vocab_size,
                                    distribution=args.distribution)
    rows = torch.cat([torch.load(path)["logits"] for path in
                      sorted(Path(args.captures).glob("*/logits-*.pt"))]).cuda().float()
    index = torch.arange(config.batch_size, device="cuda") % rows.shape[0]
    return rows[index].contiguous()


def kernel_time(fn, source: torch.Tensor, iters: int) -> float:
    """Same measurement as benchmark_topk_topp.py: CUPTI span incl. clone, cold L2."""
    times = bench_gpu_time_with_cupti(lambda: fn(source.clone()), dry_run_iters=5,
                                      repeat_iters=iters, use_cuda_graph=False,
                                      cold_l2_cache=True)
    return round(1000 * sum(times) / len(times), 1)


if __name__ == "__main__":
    main()
