"""gate.py with a kernel copy from this folder registered as impl `new` (main's Triton stays `triton`).

The module is $NEW_MODULE (default topk_topp_triton, base_triton is the unmodified kernel).
"""

import importlib
import os
import sys

import gate


def register(bench, impl: str) -> None:
    module = importlib.import_module(os.environ.get("NEW_MODULE", "topk_topp_triton"))
    bench.IMPLEMENTATIONS["new"] = module.apply_top_k_top_p_triton


if __name__ == "__main__":
    gate.register_extra = register
    sys.argv += ["--impl", "new"]
    gate.main()
