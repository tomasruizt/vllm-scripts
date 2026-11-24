"""Main's Triton kernel ("triton", from the installed vLLM) vs. the PR's ("new", loaded from a
file) in one process, alternating per case, on the fixed-timing harness (bench_fixed) with
the exact check. For images without the CUDA kernel PR (e.g. the Vera Rubin vLLM image).

Usage: python vr_ab.py NEW_KERNEL.py BENCH_ARGS...   (BENCH_ARGS go to bench_fixed's CLI)
"""

import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent


def main() -> None:
    stub_cuda_module()
    bench = load_module("benchmark_topk_topp", HERE / "bench_fixed/benchmark_topk_topp.py")
    new = load_module("vllm.v1.sample.ops.topk_topp_triton_new", Path(sys.argv[1]))
    bench.IMPLEMENTATIONS["new"] = new.apply_top_k_top_p_triton
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    bench.main()


def stub_cuda_module() -> None:
    """bench_fixed imports the CUDA kernel PR's module, which this image lacks."""
    stub = types.ModuleType("vllm.v1.sample.ops.topk_topp_cuda")
    stub.apply_top_k_top_p_cuda = None
    stub.topk_topp_cuda_available = lambda: False
    sys.modules[stub.__name__] = stub


def load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
