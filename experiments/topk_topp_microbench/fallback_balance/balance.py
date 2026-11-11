"""PR kernel (pr_kernel.py) vs. load-balanced fallbacks (NAME_kernel.py), fixed timing.

Usage, on the GPU host (cwd holds the kernels, gate.py, large_k.py, nv_after_rocm.py):
    python balance.py OUT_DIR --qwen-rows Q.pt [--glm-rows G.pt] --suite gate|tiles|report [--passes N]
        [--impls pr fix ...] [--only SUBSTR ...]
gate: the 19 gate cases. tiles: real top-p, Qwen3.6 rows tiled vs. shuffled, b256 to b2048.
report: the 112 report cases plus GLM real top-p. Each case also records whether every
impl's masked logits equal the first's bit for bit.
"""

import argparse
import importlib
import json
from pathlib import Path

import torch
from gate import gate_configs
from large_k import load_benchmark
from nv_after_rocm import BATCHES, report_configs, stub_cuda_filter

IMPLS = ["pr", "fix"]
TILE_BATCHES = [256, 512, 1024, 2048]
ONLY = [""]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--qwen-rows", required=True)
    parser.add_argument("--glm-rows")
    parser.add_argument("--suite", choices=["gate", "tiles", "report"], required=True)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--benchmark", default="benchmark_topk_topp.py")
    parser.add_argument("--impls", nargs="+", default=IMPLS)
    parser.add_argument("--only", nargs="+", help="keep cases whose name holds one of these")
    args = parser.parse_args()
    IMPLS[:] = args.impls
    ONLY[:] = args.only or [""]
    args.out.mkdir(parents=True, exist_ok=True)
    stub_cuda_filter()
    bench = load_benchmark(Path(args.benchmark))
    register(bench)
    (args.out / "gpu.txt").write_text(torch.cuda.get_device_name() + "\n")
    qwen = torch.load(args.qwen_rows).float().cuda()
    for i in range(args.passes):
        run_suite(bench, args, qwen, args.out / f"{args.suite}_r{i}.jsonl")
    print("ALL_DONE", flush=True)


def run_suite(bench, args, qwen: torch.Tensor, output: Path) -> None:
    output.unlink(missing_ok=True)
    if args.suite == "gate":
        run(bench, output, gate_configs(bench, qwen), qwen, "qwen")
    elif args.suite == "tiles":
        tiles(bench, output, qwen)
    else:
        run(bench, output, report_configs(bench, qwen), qwen, "qwen")
        glm_rows = torch.load(args.glm_rows).float().cuda()
        configs = bench.create_real_configs(BATCHES, glm_rows.shape[1])
        run(bench, output, [c for c in configs if c.name.startswith("real_topp_")], glm_rows, "glm")


def tiles(bench, output: Path, qwen: torch.Tensor) -> None:
    """Real top-p with the rows tiled (the hard rows every 16) and shuffled (seed 0)."""
    configs = {c.batch_size: c for c in bench.create_real_configs(TILE_BATCHES, qwen.shape[1])
               if c.name.startswith("real_topp_")}  # fmt: skip
    for batch, config in configs.items():
        tiled = bench.tile_rows(qwen, batch)
        order = torch.randperm(batch, generator=torch.Generator().manual_seed(0)).cuda()
        for layout, rows in (("tiled", tiled), ("shuffled", tiled[order].contiguous())):
            run(bench, output, [config], rows, "qwen", layout)


def run(bench, output: Path, configs: list, rows: torch.Tensor, source: str, layout: str = "tiled") -> None:
    """Time and check each case, then tag its rows with the mask comparison."""
    for config in [c for c in configs if any(s in c.name for s in ONLY)]:
        tmp = output.with_suffix(".tmp")
        tmp.unlink(missing_ok=True)
        bench.run_benchmark([config], IMPLS, check=True, output=tmp, real_rows=rows, verbose=False)
        row = json.loads(tmp.read_text()) | {"Source": source, "Layout": layout, "Case": config.name}
        row["masks equal"] = masks_equal(bench, config, rows)
        with output.open("a") as file:
            file.write(json.dumps(row) + "\n")
        times = " ".join(f"{name}={1000 * row[f'{name} (ms)']:.1f}us" for name in IMPLS)
        checks = all(row[f"{name} check"] for name in IMPLS)
        print(f"{config.name} {layout}: {times} check={checks} equal={row['masks equal']}", flush=True)


def masks_equal(bench, config, rows: torch.Tensor) -> bool:
    logits = bench.tile_rows(rows, config.batch_size) if config.real_logits else bench.create_logits(
        config.batch_size, config.vocab_size)  # fmt: skip
    k, p = config.k_values, config.p_values
    outs = [bench.IMPLEMENTATIONS[name](logits.clone(), k, p) for name in IMPLS]
    return all(torch.equal(outs[0], out) for out in outs[1:])


def register(bench) -> None:
    for name in IMPLS:
        if name in bench.IMPLEMENTATIONS:  # e.g. triton: the installed vLLM's kernel
            continue
        module = importlib.import_module(f"{name}_kernel")
        bench.IMPLEMENTATIONS[name] = module.apply_top_k_top_p_triton


if __name__ == "__main__":
    main()
