"""Count which CUDA filter paths each benchmark scenario takes.

Needs a build with stats.patch applied: the kernel adds per-row path counters in
the last 256 bytes of its workspace. Usage, from the PR tree:
python path_counts.py OUT.jsonl --logits-file ROWS.pt [--batch-sizes ...]
"""

import argparse
import json
from pathlib import Path

import torch
from large_k import large_k_configs, load_benchmark

from vllm.v1.sample.ops import topk_topp_cuda

EVENTS = [
    "rows", "restart", "overflow", "global_large_k", "global_stream", "slow_p",
    "second_gather_try", "second_gather_ok", "window_try", "window_ok",
    "global_after_window", "mask_row", "scatter", "topk_select", "noop",
]  # fmt: skip


def main() -> None:
    args = parse_args()
    bench = load_benchmark(Path("benchmarks/benchmark_topk_topp.py"))
    real_rows = torch.load(args.logits_file).float().cuda()
    configs = bench.create_benchmark_configs(args.batch_sizes, args.vocab_sizes)
    configs += bench.create_real_configs(args.batch_sizes, real_rows.shape[1])
    configs += large_k_configs(bench, args.batch_sizes)
    with open(args.output, "w") as out:
        for config in configs:
            out.write(json.dumps(count(bench, config, real_rows)) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--logits-file", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=[1, 4, 8, 16, 64, 128, 256, 512, 1024, 2048])  # fmt: skip
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[131072, 248320])
    return parser.parse_args()


def count(bench, config, real_rows: torch.Tensor) -> dict:
    if config.real_logits:
        logits = bench.tile_rows(real_rows, config.batch_size)
    else:
        logits = bench.create_logits(config.batch_size, config.vocab_size)
    stats = topk_topp_cuda._workspace(logits.device)[-256:].view(torch.int64)
    stats.zero_()
    topk_topp_cuda.apply_top_k_top_p_cuda(logits, config.k_values, config.p_values)
    values = stats[: len(EVENTS)].tolist()
    scenario = config.name.split("_b")[0]
    return dict(scenario=scenario, batch=config.batch_size, vocab=config.vocab_size,
                **dict(zip(EVENTS, values)))  # fmt: skip


if __name__ == "__main__":
    main()
