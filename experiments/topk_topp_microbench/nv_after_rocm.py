"""Main's Triton vs. the previous (prev_kernel) and the ROCm-ported (cand_kernel) top-k/top-p
kernel in one process, checked against the exact reference, plus clone-only times.

Usage, from a tree with main's Triton and the unified benchmark (or an image with both):
    python nv_after_rocm.py OUT_DIR --qwen-rows Q.pt [--glm-rows G.pt] [--gate-runs N]
Without --gate-runs: the 112 report cases (+ the GLM real_topp rows). With it: the 19-case gate N times.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch
from gate import gate_configs
from large_k import load_benchmark

IMPLS = ["triton", "prev", "cand"]  # set by --impls
BATCHES = [8, 16, 64, 128, 256, 512, 1024, 2048]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--qwen-rows", required=True)
    parser.add_argument("--glm-rows")
    parser.add_argument("--gate-runs", type=int, default=0)
    parser.add_argument("--impls", nargs="+", default=IMPLS, help="triton or NAME for NAME_kernel.py")
    parser.add_argument("--benchmark", default="benchmarks/benchmark_topk_topp.py")
    parser.add_argument("--no-clone", action="store_true", help="skip clone-only times (fixed timing)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stub_cuda_filter()
    bench = load_benchmark(Path(args.benchmark))
    for name in set(args.impls) - {"triton"}:
        bench.IMPLEMENTATIONS[name] = importlib.import_module(f"{name}_kernel").apply_top_k_top_p_triton
    IMPLS[:] = args.impls
    (args.out / "gpu.txt").write_text(torch.cuda.get_device_name() + "\n")
    qwen = torch.load(args.qwen_rows).float().cuda()
    if args.gate_runs:
        shapes = [run(bench, args.out / f"gate_r{i}.jsonl", gate_configs(bench, qwen), qwen) for i in range(args.gate_runs)][0]
    else:
        shapes = run(bench, args.out / "report.jsonl", report_configs(bench, qwen), qwen)
        shapes |= glm(bench, args.out, args.glm_rows) if args.glm_rows else set()
    if not args.no_clone:
        clone(args.out / "clone.jsonl", sorted(shapes))
    print("ALL_DONE", flush=True)


def stub_cuda_filter() -> None:
    """In a stock image, the benchmark's CUDA-filter import comes from a local copy (topk_topp_cuda.py)."""
    try:
        import vllm.v1.sample.ops.topk_topp_cuda  # noqa: F401
    except ModuleNotFoundError:
        import topk_topp_cuda

        sys.modules["vllm.v1.sample.ops.topk_topp_cuda"] = topk_topp_cuda


def report_configs(bench, qwen: torch.Tensor) -> list:
    configs = bench.create_benchmark_configs(BATCHES, [131072, 248320])
    return configs + bench.create_real_configs(BATCHES, qwen.shape[1])


def glm(bench, out: Path, path: str) -> set:
    rows = torch.load(path).float().cuda()
    configs = [c for c in bench.create_real_configs(BATCHES, rows.shape[1]) if c.name.startswith("real_topp_")]
    shapes = run(bench, out / "glm.jsonl", configs, rows)
    del rows
    return shapes


def run(bench, output: Path, configs: list, rows: torch.Tensor) -> set:
    output.unlink(missing_ok=True)
    bench.run_benchmark(configs, IMPLS, check=True, output=output, real_rows=rows)
    return {(c.batch_size, c.vocab_size) for c in configs}


def clone(output: Path, shapes: list) -> None:
    """Clone-only time, timed like the benchmark (CUPTI, cold L2, 5 warmup + 20 timed)."""
    from flashinfer.testing import bench_gpu_time_with_cupti

    with output.open("w") as file:
        for batch, vocab in shapes:
            logits = torch.randn(batch, vocab, device="cuda")
            times = bench_gpu_time_with_cupti(lambda: logits.clone(), dry_run_iters=5, repeat_iters=20,
                                              use_cuda_graph=False, cold_l2_cache=True)  # fmt: skip
            file.write(json.dumps({"Batch": batch, "Vocab": vocab, "clone-only (ms)": sum(times) / len(times)}) + "\n")


if __name__ == "__main__":
    main()
