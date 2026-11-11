"""Clone-only time per (batch, vocab), timed like benchmark_topk_topp.py (CUPTI, cold L2,
5 warmup + 20 timed calls, mean)."""
import json
import sys

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

out = open(sys.argv[1], "w")
for vocab in [int(v) for v in sys.argv[2:]] or (131072, 248320):
    for batch in (8, 16, 64, 128, 256, 512, 1024, 2048):
        logits = torch.randn(batch, vocab, device="cuda")
        times = bench_gpu_time_with_cupti(lambda: logits.clone(), dry_run_iters=5, repeat_iters=20,
                                          use_cuda_graph=False, cold_l2_cache=True)
        row = {"Batch": batch, "Vocab": vocab, "clone-only (ms)": sum(times) / len(times)}
        out.write(json.dumps(row) + "\n")
        print(row, flush=True)
print("EXIT 0")
