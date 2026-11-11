"""Fast gate for top-k/top-p kernel changes: ~19 cases, one pass, a few minutes.

Real rows plus the known worst cases (huge top-p nuclei, large k with top-p), each
checked against the exact reference and timed against main's Triton kernel. A case
fails if it is inexact or >=10% slower than Triton. With --baseline (a previous gate
JSONL), it also prints the change vs. that run, so real rows can be held within ~3%.

Usage, from a PR tree on the GPU host:
    python gate.py OUT.jsonl --logits-file ROWS.pt [--impl cuda] [--baseline OLD.jsonl]
--impl takes any name in the benchmark's registry, or hier / triton-core / cute when
their modules are importable (PYTHONPATH).
"""

import argparse
import json
from pathlib import Path

import torch
from large_k import config as large_k_config
from large_k import load_benchmark

REAL = [("real_topk_topp", b) for b in (8, 256, 2048)] + [("real_topp", b) for b in (8, 64, 128, 256, 2048)]
SYNTH = [("topp_whole", 1, 248320), ("topp_whole", 8, 248320), ("topp_whole", 256, 248320),
         ("topp_partial", 8, 248320), ("topp_partial", 256, 131072),
         ("topp_whole", 256, 131072),
         ("topk_topp_whole", 8, 248320), ("topk_topp_whole", 1024, 248320)]  # fmt: skip
LARGE_K = [(10000, 0.9, 256), (10000, 0.9, 1024), (1000, None, 8)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--logits-file", required=True)
    parser.add_argument("--impl", default="cuda")
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    bench = load_benchmark(Path("benchmarks/benchmark_topk_topp.py"))
    register_extra(bench, args.impl)
    real_rows = torch.load(args.logits_file).float().cuda()
    args.output.unlink(missing_ok=True)
    bench.run_benchmark(gate_configs(bench, real_rows), ["triton", args.impl], check=True,
                        output=args.output, real_rows=real_rows, verbose=False)  # fmt: skip
    report(args.output, args.impl, args.baseline)


def gate_configs(bench, real_rows: torch.Tensor) -> list:
    vocab = real_rows.shape[1]
    real = {c.name: c for c in bench.create_real_configs([8, 64, 128, 256, 2048], vocab)}
    synth = {c.name: c for c in bench.create_benchmark_configs([1, 8, 256, 1024], [131072, 248320])}
    configs = [real[f"{s}_b{b}_v{vocab // 1000}k"] for s, b in REAL]
    configs += [synth[f"{s}_b{b}_v{v // 1000}k"] for s, b, v in SYNTH]
    return configs + [large_k_config(bench, b, k, p) for k, p, b in LARGE_K]


def register_extra(bench, impl: str) -> None:
    """Research implementations that are not in the benchmark's registry."""
    if impl in bench.IMPLEMENTATIONS:
        return
    if impl == "triton-core":
        from triton_core_bench import register_core

        register_core(bench)
    elif impl == "hier":
        from hier_cuda import apply_top_k_top_p_hier

        bench.IMPLEMENTATIONS["hier"] = apply_top_k_top_p_hier
    elif impl == "triton-hier":
        from vllm.v1.sample.ops.topk_topp_triton_hier import (
            apply_top_k_top_p_triton_hier,
        )

        bench.IMPLEMENTATIONS["triton-hier"] = apply_top_k_top_p_triton_hier


def report(output: Path, impl: str, baseline: Path | None) -> None:
    rows = [json.loads(line) for line in output.open()]
    old = {r["Scenario"] + str(r["Batch"]) + str(r["Vocab"]): r for r in load(baseline)}
    failed = 0
    print(f"{'case':<28}{'triton µs':>11}{impl + ' µs':>12}{'vs triton':>11}{'vs base':>9}  ok")
    for r in rows:
        t, x = 1000 * r["triton (ms)"], 1000 * r[f"{impl} (ms)"]
        ok = r[f"{impl} check"] and x < 1.1 * t
        failed += not ok
        prev = old.get(r["Scenario"] + str(r["Batch"]) + str(r["Vocab"]))
        vs_base = f"{100 * (x / (1000 * prev[f'{impl} (ms)']) - 1):+.0f}%" if prev else ""
        name = f"{r['Scenario']} b{r['Batch']} v{r['Vocab'] // 1000}k"
        print(f"{name:<28}{t:>11.1f}{x:>12.1f}{100 * (x / t - 1):>+10.0f}%{vs_base:>9}  {'ok' if ok else 'FAIL'}")
    print("GATE", "PASS" if failed == 0 else f"FAIL ({failed} cases)")


def load(path: Path | None) -> list[dict]:
    return [json.loads(line) for line in path.open()] if path else []


if __name__ == "__main__":
    main()
