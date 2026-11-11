"""Launch the unified filter once per selected case, for ncu profiling."""

import argparse
import importlib

import torch

from benchmark_unified2 import load_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--kernel-module", default="unified_topk_topp_v8")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--vocab", type=int, default=65536)
    parser.add_argument("--scenarios", default="topk_whole,topp_whole")
    parser.add_argument("--distribution", default="zipf")
    args = parser.parse_args()
    prototype = importlib.import_module(args.kernel_module).UnifiedFilter()
    module = load_benchmark()
    configs = module.create_benchmark_configs([args.batch], [args.vocab], top_k=20, top_p=0.8)
    wanted = args.scenarios.split(",")
    for config in configs:
        if config.name.split("_b")[0] not in wanted:
            continue
        source = module.create_logits(config.batch_size, config.vocab_size,
                                      distribution=args.distribution)
        prototype(source.clone(), config.k_values, config.p_values)
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
