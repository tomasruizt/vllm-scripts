"""Time every Triton compile of the core kernels while running another script.

Usage: TRITON_CACHE_DIR=$(mktemp -d) python triton_core_compile.py SCRIPT.py ARGS...
Prints one line per compiled variant (kernel, seconds, constexprs that vary) and a total.
"""

import atexit
import runpy
import sys
import time

import triton_core

COMPILES: list[tuple[str, float, dict]] = []
SHOWN = ("HAS_K", "HAS_P", "SPLITS", "HSPLITS", "FSPLITS", "SUPERGROUPS_P2", "TIMING")


def main() -> None:
    for name in ("_levels_kernel", "_filter_kernel"):
        timed(getattr(triton_core, name), name)
    atexit.register(summary)
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")


def timed(kernel, name: str) -> None:
    compile_ = kernel._do_compile

    def wrapper(key, signature, device, constexprs, options, attrs, warmup):
        start = time.perf_counter()
        out = compile_(key, signature, device, constexprs, options, attrs, warmup)
        shown = {arg_name(kernel, k): v for k, v in constexprs.items()}
        shown = {k: v for k, v in shown.items() if k in SHOWN or not k.isupper()}
        COMPILES.append((name, time.perf_counter() - start, shown))
        print(f"COMPILE {name} {COMPILES[-1][1]:.1f}s {shown}", flush=True)
        return out

    kernel._do_compile = wrapper


def arg_name(kernel, key) -> str:
    return kernel.arg_names[key[0]] if isinstance(key, tuple) and len(key) == 1 else str(key)


def summary() -> None:
    for name in ("_levels_kernel", "_filter_kernel"):
        times = [t for n, t, _ in COMPILES if n == name]
        print(f"COMPILE_TOTAL {name} variants={len(times)} seconds={sum(times):.0f} max={max(times, default=0):.0f}")


if __name__ == "__main__":
    main()
