"""Zipf rows like the benchmark's synthetic cases: python make_synth.py OUT.pt ROWS VOCAB"""
import sys
from pathlib import Path

import torch
from large_k import load_benchmark

bench = load_benchmark(Path("benchmark_topk_topp.py"))
torch.save(bench.create_logits(int(sys.argv[2]), int(sys.argv[3])).cpu(), sys.argv[1])
