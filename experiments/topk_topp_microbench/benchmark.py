"""Reproduce Qwen sampling latency without loading a model or timing masked inputs."""

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import warnings
from functools import partial
from importlib.metadata import version
from pathlib import Path

import flashinfer
import torch
from flashinfer.testing import bench_gpu_time_with_cuda_event, bench_gpu_time_with_cupti

from vllm.v1.sample.ops import topk_topp_triton as native
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 9, 32])
    parser.add_argument("--vocab", type=int, default=248320)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--implementations", nargs="+", default=["native", "flashinfer-fastpath"])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("error", message=".*Falling back to CUDA events.*")
    helper = load_benchmark_helpers()
    torch.manual_seed(42)
    source = Path(native.__file__).resolve()
    metadata = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                    source=str(source), commit=subprocess.check_output(
                        ["git", "-C", str(source.parent), "rev-parse", "HEAD"], text=True).strip(),
                    command=sys.argv, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    timing_method="CUDA events, CUPTI spans, PyTorch kernel activities", cupti_python=version("cupti-python"),
                    flashinfer=flashinfer.__version__,
                    k=20, p=0.8, temperature=0.7, dtype="float32", synthetic_logits=True)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    if args.check_only:
        check_edge_cases(args, helper)
        (args.output / "DONE").write_text("ok\n")
        return
    rows = []
    for repeat in range(args.rounds):
        rows.extend(benchmark_round(args, helper, repeat))
        (args.output / "metrics.json").write_text(json.dumps(rows, indent=2))
    (args.output / "DONE").write_text("ok\n")


def benchmark_round(args: argparse.Namespace, helper: object, repeat: int) -> list[dict]:
    rows = []
    for batch in args.batches:
        for distribution in ("zipf", "normal"):
            rows.extend(benchmark_case(args, helper, batch, distribution, repeat))
    return rows


def benchmark_case(args: argparse.Namespace, helper: object, batch: int, distribution: str, repeat: int = 0) -> list[dict]:
    source = helper.create_logits(batch, args.vocab) if distribution == "zipf" else torch.randn(batch, args.vocab, device="cuda")
    source /= 0.7
    k = torch.full((batch,), 20, dtype=torch.int32, device="cuda")
    p = torch.full((batch,), 0.8, dtype=torch.float32, device="cuda")
    reference = apply_top_k_top_p_pytorch(source.clone(), k, p).softmax(-1)
    work = torch.empty_like(source)
    rows = []
    shift = repeat % len(args.implementations)
    order = args.implementations[shift:] + args.implementations[:shift]
    for name in order:
        call = implementation(name, helper, source, k, p)
        actual = call(source.clone()).softmax(-1)
        max_tv = float((actual - reference).abs().sum(-1).max()) / 2
        assert torch.allclose(actual, reference, atol=1e-6, rtol=1e-5), (name, batch, distribution, max_tv)
        measured = lambda: call(work.copy_(source))
        eager = bench_gpu_time_with_cuda_event(measured, dry_run_iters=5,
            repeat_iters=args.repeats, cold_l2_cache=False)
        times = bench_gpu_time_with_cupti(measured, dry_run_iters=5,
            repeat_iters=args.repeats, use_cuda_graph=False, cold_l2_cache=False)
        trace_path = args.output / f"b{batch}-{distribution}-{name}-r{repeat}.json"
        kernel_times = profile_kernels(measured, trace_path)
        row = dict(batch=batch, vocab=args.vocab, distribution=distribution, round=repeat,
                   implementation=name, max_tv=max_tv, allclose=True,
                   eager_us=[float(t) * 1000 for t in eager],
                   eager_median_us=statistics.median(eager) * 1000,
                   cupti_us=[float(t) * 1000 for t in times],
                   cupti_median_us=statistics.median(times) * 1000,
                   kernel_us=kernel_times,
                   gpu_busy_mean_us=sum(sum(v) for v in kernel_times.values()) / 20,
                   topk_topp_median_us=statistics.median(kernel_times.get("_topk_topp_kernel", [])) if "_topk_topp_kernel" in kernel_times else None)
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("cupti_us", "eager_us", "kernel_us")}), flush=True)
    return rows


def implementation(name: str, helper: object, source: torch.Tensor, k: torch.Tensor, p: torch.Tensor):
    if name == "native":
        return lambda logits: native.apply_top_k_top_p_triton(logits, k, p, has_topp_only_rows=False)
    if name == "flashinfer-fastpath":
        kwargs = helper.prepare_kwargs(name, k, p, source.shape[-1])
        return lambda logits: helper.apply_top_k_top_p_flashinfer_fastpath(logits, k, p, **kwargs)
    if name == "pytorch-compact":
        return lambda logits: compact_filter(logits, torch.topk, p)
    if name == "flashinfer-compact":
        return lambda logits: compact_filter(logits, flashinfer.top_k, p)
    if name == "flashinfer-deterministic":
        select = partial(flashinfer.top_k, deterministic=True, tie_break=1)
        return lambda logits: compact_filter(logits, select, p)
    raise ValueError(name)


def compact_filter(logits: torch.Tensor, select, p: torch.Tensor | None, top_k: int = 20) -> torch.Tensor:
    logits = logits.contiguous()
    values, indices = select(logits, top_k, sorted=True)
    values = values.flip(-1)
    indices = indices.flip(-1)
    if p is not None:
        remove = values.softmax(-1).cumsum(-1) <= 1 - p.unsqueeze(-1)
        remove[:, -1] = False
        values = values.masked_fill(remove, -float("inf"))
    output = torch.full_like(logits, -float("inf"))
    return output.scatter_(1, indices, values)


def check_edge_cases(args: argparse.Namespace, helper: object) -> None:
    rows = []
    for label, source, top_p in edge_cases(args.vocab):
        k = torch.full((source.shape[0],), 20, dtype=torch.int32, device="cuda")
        p = torch.full((source.shape[0],), top_p, device="cuda")
        reference = apply_top_k_top_p_pytorch(source.clone(), k, p).softmax(-1)
        original = implementation("native", helper, source, k, p)(source.clone()).softmax(-1)
        for name in args.implementations:
            actual = implementation(name, helper, source, k, p)(source.clone()).softmax(-1)
            finite = bool(torch.isfinite(reference).all())
            row = dict(case=label, implementation=name, p=top_p,
                       matches_pytorch=bool(torch.allclose(actual, reference, atol=1e-6, rtol=1e-5, equal_nan=True)),
                       matches_native=bool(torch.allclose(actual, original, atol=1e-6, rtol=1e-5, equal_nan=True)),
                       max_tv=float((actual-reference).abs().sum(-1).max()) / 2 if finite else None)
            rows.append(row)
            print(json.dumps(row), flush=True)
    (args.output / "edge-cases.json").write_text(json.dumps(rows, indent=2))


def edge_cases(vocab: int) -> list[tuple[str, torch.Tensor, float]]:
    rounded = torch.randn(4, vocab, device="cuda").bfloat16().float() / 0.7
    sparse = torch.full((4, 256), -float("inf"), device="cuda")
    sparse[:, :13] = torch.randn(4, 13, device="cuda")
    boundary = torch.full((4, 256), -10.0, device="cuda")
    boundary[:, :19] = torch.linspace(3, 2, 19, device="cuda")
    boundary[:, 19:22] = 1.0
    nucleus = torch.full((4, 256), -float("inf"), device="cuda")
    nucleus[:, :3] = torch.tensor([0.5, 0.25, 0.25], device="cuda").log()
    return [("bf16-rounded", rounded, 0.8), ("sparse", sparse, 0.8),
            ("topk-ties", boundary, 1.0), ("topp-boundary", nucleus, 0.75),
            ("all-masked", torch.full_like(sparse, -float("inf")), 0.8)]


def profile_kernels(call, output: Path) -> dict[str, list[float]]:
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as profiler:
        for _ in range(20):
            call()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(output))
    kernels = {}
    for event in json.loads(output.read_text())["traceEvents"]:
        if event.get("cat") == "kernel" and event.get("ph") == "X":
            kernels.setdefault(event["name"], []).append(event["dur"])
    return kernels


def load_benchmark_helpers():
    path = Path(native.__file__).resolve().parents[4] / "benchmarks/benchmark_topk_topp.py"
    spec = importlib.util.spec_from_file_location("existing_topk_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
