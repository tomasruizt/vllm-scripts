"""Run one masking call per case (after a warmup call) to emit device traces.

Usage: trace_probe.py scenario:batch:vocab[:p] ...
"""

import os
import sys

import torch
from kernel_times import config, create_logits

from vllm.v1.sample.ops.topk_topp_cuda import apply_top_k_top_p_cuda

WARM_CODE = os.environ.get("WARM_CODE") == "1"

for case in sys.argv[1:]:
    scenario, batch, vocab, *p_override = case.split(":")
    c = config(scenario, int(batch), int(vocab))
    logits = create_logits(c.batch_size, c.vocab_size)
    p = c.p_values
    if p_override:
        p = torch.full_like(p, float(p_override[0]))
    apply_top_k_top_p_cuda(logits.clone(), c.k_values, p)
    torch.cuda.synchronize()
    torch.empty(512 << 20, dtype=torch.uint8, device="cuda").zero_()
    if WARM_CODE:
        tk, tp = (None if t is None else t[:1] for t in (c.k_values, p))
        apply_top_k_top_p_cuda(logits[:1].clone(), tk, tp)
    torch.cuda.synchronize()
    print(f"CASE {case}", flush=True)
    apply_top_k_top_p_cuda(logits.clone(), c.k_values, p)
    torch.cuda.synchronize()
