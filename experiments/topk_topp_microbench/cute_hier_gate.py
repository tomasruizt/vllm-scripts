"""Fast gate for the CuTe hierarchical kernel (rh-4spj): a dozen representative cases, one pass, one
compile, in a few minutes. Exact check on every case plus edge cases, then pass/fail against the
acceptance criteria: no case >= 10% slower than Triton, real rows within 3% of the best version so far.

Usage, from the CuTe tree: python cute_hier_gate.py OUT.jsonl --logits-file ROWS.pt [--best BEST.jsonl]
BEST.jsonl is a previous gate output (or benchmark jsonl) with "<GATE_IMPL> (ms)", GATE_IMPL defaults to cute-hier.
"""

import argparse
import json
from pathlib import Path

import torch
from large_k import config as large_k_config
from large_k import load_benchmark

# (scenario, batch, vocab): real rows, the known worst cases (Zipf top-p with huge nuclei), large k.
CASES = [
    ("real_topk_topp", 1, 248320), ("real_topk_topp", 64, 248320), ("real_topk_topp", 2048, 248320),
    ("real_topp", 1, 248320), ("real_topp", 64, 248320), ("real_topp", 2048, 248320),
    ("topp_whole", 1, 248320), ("topp_whole", 16, 248320), ("topp_whole", 4, 131072),
    ("topp_whole", 64, 131072), ("topp_partial", 256, 131072), ("topk_whole", 16, 248320),
]  # fmt: skip
LARGE_K = [(256, 10000, 0.9), (8, 4096, None)]
# Edge cases of tests/v1/sample/test_topk_topp_sampler.py: (kind, batch, vocab, k, p, splits).
EDGES = [
    ("bf16", 64, 65536, "mixed", "mixed", 0), ("zeros", 8, 65536, 20, 0.5, 0),
    ("huge", 4, 131072, None, 0.95, 0), ("normal", 3, 65537, "mixed", "mixed", 0),
    ("normal", 8, 131072, 5000, 0.9, 0), ("normal", 8, 65536, 30000, None, 0),
    ("zipf", 4, 131072, 20, 0.8, 1), ("zipf", 4, 131072, 20, 0.8, 16), ("normal", 1, 131072, None, 0.9, 16),
]  # fmt: skip
REAL = ("real_topk_topp", "real_topp")
IMPL = __import__("os").environ.get("GATE_IMPL", "cute-hier")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--logits-file", required=True)
    parser.add_argument("--best", type=Path, help="earlier gate or benchmark jsonl")
    args = parser.parse_args()
    edges_ok = check_edges()
    bench = load_benchmark(Path("benchmarks/benchmark_topk_topp.py"))
    real_rows = torch.load(args.logits_file).float().cuda()
    configs = [find(bench, real_rows, *case) for case in CASES]
    configs += [large_k_config(bench, b, k, p) for b, k, p in LARGE_K]
    args.output.unlink(missing_ok=True)
    bench.run_benchmark(configs, [IMPL, "triton"], check=True, output=args.output, real_rows=real_rows)
    verdict(load(args.output), load(args.best) if args.best else {}, edges_ok)


def find(bench, real_rows, scenario: str, batch: int, vocab: int):
    if scenario in REAL:
        configs = bench.create_real_configs([batch], real_rows.shape[1])
    else:
        configs = bench.create_benchmark_configs([batch], [vocab])
    return next(c for c in configs if c.name.startswith(f"{scenario}_b{batch}_"))


def check_edges() -> bool:
    from tests.v1.sample.test_topk_topp_sampler import _assert_exact_filter, _make_logits, _make_param

    import vllm.v1.sample.ops.topk_topp_cute as th

    torch.set_default_device("cuda")
    choose = th._Plan.choose_splits
    gen = torch.Generator(device="cuda").manual_seed(7)
    ok = True
    for kind, batch, vocab, k, p, splits in EDGES:
        th._Plan.choose_splits = (lambda self, b, v, s=splits: s) if splits else choose
        source = _make_logits(kind, batch, vocab, gen).cuda()
        kt = _make_param(k, batch, vocab, gen, torch.int32)
        pt = _make_param(p, batch, vocab, gen, torch.float32)
        try:
            _assert_exact_filter(source, th.apply_top_k_top_p_cute(source.clone(), kt, pt), kt, pt)
        except AssertionError as e:
            ok = False
            print("EDGE FAIL", kind, batch, vocab, k, p, splits, str(e)[:120])
    th._Plan.choose_splits = choose
    print("edges", "ok" if ok else "FAIL")
    return ok


def load(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {(r["Scenario"], r["Batch"], r["Vocab"]): r for r in rows}


def verdict(rows: dict, best: dict, edges_ok: bool) -> None:
    passed = edges_ok
    print(f"{'case':34s} {'hier µs':>9s} {'triton':>9s} {'ratio':>6s} {'best':>9s} {'vs best':>7s}  verdict")
    for key, r in rows.items():
        ours = r[f"{IMPL} (ms)"] * 1e3
        # Triton timed next to one kernel at small batch runs slow: prefer the full-suite median.
        triton = best.get(key, {}).get("triton (ms)", r["triton (ms)"]) * 1e3
        prev = best.get(key, {}).get(f"{IMPL} (ms)")
        fails = [] if r[f"{IMPL} check"] else ["inexact"]
        if ours > 1.1 * triton:
            fails.append("slower than triton")
        if prev and key[0] in REAL and ours > 1.03 * prev * 1e3:
            fails.append("real row regressed")
        passed &= not fails
        rel = f"{ours / (prev * 1e3):7.2f}" if prev else f"{'-':>7s}"
        prev_us = f"{prev * 1e3:9.1f}" if prev else f"{'-':>9s}"
        print(f"{'/'.join(map(str, key)):34s} {ours:9.1f} {triton:9.1f} {ours / triton:6.2f} {prev_us} {rel}  {', '.join(fails) or 'ok'}")
    print("GATE", "PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
