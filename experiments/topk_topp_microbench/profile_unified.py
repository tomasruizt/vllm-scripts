"""Separate GPU kernel costs from launch gaps in the eager benchmark span."""

import argparse
import importlib
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from benchmark_unified2 import load_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-module", default="unified_topk_topp_v5")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prototype = importlib.import_module(args.kernel_module).UnifiedFilter()
    module = load_benchmark()
    configs = module.create_benchmark_configs([512], [65536], top_k=20, top_p=0.8)
    rows = []
    for config in configs:
        source = module.create_logits(512, 65536, distribution="zipf")
        for name, fn in [("triton", module.IMPLEMENTATIONS["triton"]), ("unified", prototype)]:
            rows.extend(capture(args.output, config, source, name, fn))
    (args.output / "kernels.json").write_text(json.dumps(rows, indent=2))
    (args.output / "DONE").write_text("ok\n")


def capture(output: Path, config, source: torch.Tensor, name: str, fn) -> list[dict]:
    def run() -> None:
        fn(source.clone(), config.k_values, config.p_values)

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function(f"{config.name}/{name}"):
            run()
        torch.cuda.synchronize()
    path = output / f"{config.name}-{name}.json"
    prof.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    rows = [dict(scenario=config.name, implementation=name, kernel=e["name"], us=e["dur"])
            for e in events if e.get("cat") == "kernel" and "dur" in e]
    print(json.dumps(dict(scenario=config.name, implementation=name, kernels=rows)), flush=True)
    return rows


if __name__ == "__main__":
    main()
