"""Startup cost of the CuTe DSL filter: JIT compile of its kernels (empty or warm
CUTE_DSL_CACHE_DIR), and the first call after compilation vs. the steady state.
Usage, from the CuTe tree: python cute_startup.py OUT.json (one process per measurement)."""

import json
import sys
import time

import torch

t0 = time.perf_counter()
from vllm.v1.sample.ops.topk_topp_cute import apply_top_k_top_p_cute, reserve_topk_topp_cute_workspace  # noqa: E402

t_import = time.perf_counter() - t0
torch.zeros(1, device="cuda")
torch.cuda.synchronize()
t1 = time.perf_counter()
reserve_topk_topp_cute_workspace(torch.device("cuda"))
torch.cuda.synchronize()
t_compile = time.perf_counter() - t1


def timed_call(batch: int) -> float:
    logits = torch.randn(batch, 248320, device="cuda")
    p = torch.full((batch,), 0.9, device="cuda")
    torch.cuda.synchronize()
    t = time.perf_counter()
    apply_top_k_top_p_cute(logits, None, p)
    torch.cuda.synchronize()
    return time.perf_counter() - t


first = {b: timed_call(b) for b in (1, 256)}
steady = {b: min(timed_call(b) for _ in range(20)) for b in (1, 256)}
out = dict(import_s=t_import, compile_s=t_compile, **{f"first_b{b}_ms": v * 1e3 for b, v in first.items()},
           **{f"steady_b{b}_ms": v * 1e3 for b, v in steady.items()})
json.dump(out, open(sys.argv[1], "w"))
print(json.dumps(out))
