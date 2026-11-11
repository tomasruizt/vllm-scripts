"""CPU-only cold compile time of the New Triton kernels, and the variants a gate run needs.

No GPU is used: the kernels are compiled for a fixed target (default sm100, B200).
Usage: CUDA_VISIBLE_DEVICES= python triton_core_ctime.py [--rows 8] [--vocab 248320] [--arch 100]
"""

import argparse
import time
from pathlib import Path

import triton
import triton_core
from triton.backends.compiler import GPUTarget
from triton.knobs import runtime

# (rows, vocab, HAS_K, HAS_P) of the gate's cases
GATE = [(b, 248320, k, True) for b in (8, 256, 2048) for k in (True, False)]
GATE += [(1, 248320, False, True), (8, 248320, False, True), (256, 248320, False, True),
         (256, 131072, False, True), (8, 248320, True, True), (1024, 248320, True, True),
         (256, 248320, True, True), (1024, 248320, True, True), (8, 248320, True, False)]  # fmt: skip
POINTERS = dict(logits_ptr="fp32", level1_ptr="fp32", level2_ptr="fp32", stats_ptr="fp64",
                row_stats_ptr="fp64", state_ptr="i32", cut_ptr="i64", kept_ptr="i64",
                hist_ptr="i32", scratch_ptr="i64", window_ptr="i32",
                found_ptr="i32", more_ptr="fp64", sync_ptr="i32", k_ptr="i32", p_ptr="fp32")  # fmt: skip
STAGES: dict[str, float] = {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=248320)
    parser.add_argument("--arch", type=int, default=100)
    parser.add_argument("--sms", type=int, default=148)
    parser.add_argument("--kernels", nargs="+", default=["levels", "filter"])
    parser.add_argument("--noinline", nargs="*", default=[], help="helpers to compile as calls")
    args = parser.parse_args()
    for name in args.noinline:
        getattr(triton_core, name).noinline = True
    variants = {name: {variant_key(*args_of(name, *shape, args.sms)) for shape in GATE} for name in args.kernels}
    for name, keys in variants.items():
        print(f"VARIANTS {name} {len(keys)}")
    runtime.add_stages_inspection_hook = time_stages
    for name in args.kernels:
        compile_one(name, *args_of(name, args.rows, args.vocab, True, True, args.sms), args.arch)


def args_of(name: str, rows: int, vocab: int, has_k: bool, has_p: bool, sms: int) -> tuple:
    """(kernel, runtime ints, constexprs, launch options) of a call, as the host code builds them."""
    return triton_core.launch_args(name, rows, vocab, has_k, has_p, sms)


def variant_key(kernel, ints: dict, constexprs: dict, options: dict) -> tuple:
    """Triton compiles one variant per constexpr set and per int divisibility (or value 1)."""
    skip = {p.name for p in kernel.params if p.do_not_specialize}
    spec = tuple((k, v == 1, v % 16 == 0) for k, v in sorted(ints.items()) if k not in skip)
    return kernel.__name__, spec, tuple(sorted(constexprs.items())), tuple(sorted(options.items()))


def compile_one(name: str, kernel, ints: dict, constexprs: dict, options: dict, arch: int) -> None:
    jit = getattr(triton_core, kernel.__name__)
    signature, attrs = {}, {}
    defaults = {p.name: p.default for p in jit.params if p.name not in ints and p.name not in POINTERS}
    constexprs = {**defaults, **constexprs}
    for i, arg in enumerate(jit.arg_names):
        if arg in constexprs:
            signature[arg] = "constexpr"
        elif arg in POINTERS:
            signature[arg] = "*" + POINTERS[arg]
            attrs[(i,)] = [["tt.divisibility", 16]]
        else:
            signature[arg] = "i32"
            special = ints[arg] % 16 == 0 and not jit.params[i].do_not_specialize
            attrs.update({(i,): [["tt.divisibility", 16]]} if special else {})
    src = triton.compiler.ASTSource(jit, signature, constexprs, attrs)
    STAGES.clear()
    start = time.perf_counter()
    compiled = triton.compile(src, target=GPUTarget("cuda", arch, 32), options=options)
    stages = " ".join(f"{k}={v:.1f}" for k, v in STAGES.items())
    print(f"CTIME {name} {time.perf_counter() - start:.1f}s ({stages}) shared={compiled.metadata.shared}")
    for ext in ("ptx", "ttgir"):
        Path(f"{name}.{ext}").write_text(compiled.asm[ext])
    Path(f"{name}.cubin").write_bytes(compiled.asm["cubin"])


def time_stages(backend=None, stages: dict | None = None, *rest):
    """Triton calls the hook without arguments for a cache key, then to edit the stages."""
    if stages is None:
        return "ctime", "ctime"
    for name, fn in list(stages.items()):
        stages[name] = timed(name, fn)


def timed(name: str, fn):
    def run(src, metadata):
        start = time.perf_counter()
        out = fn(src, metadata)
        STAGES[name] = time.perf_counter() - start
        return out

    return run


if __name__ == "__main__":
    main()
