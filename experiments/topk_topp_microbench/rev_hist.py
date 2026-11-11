"""Paired A/B: the PR's top-k/top-p Triton kernel (pr_kernel.py) vs. the same kernel with the
fallback histogram pass in reverse row order (rev_kernel.py, from rev_hist.patch).

Acceptance: masks of rev identical to pr and exact vs. the FP64 reference, no case >=3% slower
than pr on any GPU, a measurable gain somewhere.
Per case: one exactness check, then PASSES passes of both impls in alternating order (AB, BA, ..),
each 5 warmup + 20 timed calls with fixed timing (input restored, 1 GiB flushed, outside the events).

Both kernels share one workspace (allocated up front): with a workspace per module, an A/A run of
two copies of the same kernel on RTX PRO 6000 differed by up to 8% per case. With a shared one, some
small cases still differed by up to 3.5%, tied to the slot (first loaded, checked and timed), so half
the processes (--proc) run with --swap (rev in the first slot), and the summary averages over them.
--aa times pr vs. a copy of pr.

Usage: python rev_hist.py OUT_DIR --qwen-rows Q.pt [--glm-rows G.pt] [--suite gate|report] [--passes N]
       [--aa] [--swap] [--proc I]
"""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from gate import gate_configs
from large_k import load_benchmark
from nv_after_rocm import stub_cuda_filter

IMPLS = ["pr", "rev"]
BATCHES = [8, 16, 64, 128, 256, 512, 1024, 2048]


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stub_cuda_filter()
    bench = load_benchmark(Path(args.benchmark))
    if args.swap:
        IMPLS.reverse()
    kernels = load_kernels(args.aa)
    for name, kernel in kernels.items():
        bench.IMPLEMENTATIONS[name] = kernel.apply_top_k_top_p_triton
    gpu = torch.cuda.get_device_name()
    meta = dict(gpu=gpu, aa=args.aa, swap=args.swap, proc=args.proc)
    qwen = torch.load(args.qwen_rows).float().cuda()
    groups = [("qwen", qwen, suite_configs(bench, args.suite, qwen))]
    if args.glm_rows and args.suite == "report":
        glm = torch.load(args.glm_rows).float().cuda()
        groups.append(("glm", glm, [c for c in bench.create_real_configs(BATCHES, glm.shape[1]) if c.name.startswith("real_topp_")]))
    share_workspace(kernels, {c.vocab_size for _, _, configs in groups for c in configs})
    for suite, rows, configs in groups:
        for i, config in enumerate(configs):
            run_case(bench, config, rows, args, meta | dict(suite=suite if config.real_logits else "synth"))
            print(f"{suite} {i + 1}/{len(configs)} {config.name}", flush=True)
    print("ALL_DONE", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--qwen-rows", required=True)
    parser.add_argument("--glm-rows")
    parser.add_argument("--suite", choices=["gate", "report"], default="gate")
    parser.add_argument("--passes", type=int, default=8)
    parser.add_argument("--benchmark", default="benchmark_topk_topp.py")
    parser.add_argument("--aa", action="store_true", help="time pr against a copy of pr")
    parser.add_argument("--swap", action="store_true", help="rev in the first slot (loaded, checked and timed first)")
    parser.add_argument("--proc", type=int, default=0, help="process index, recorded in the rows")
    return parser.parse_args()


def load_kernels(aa: bool) -> dict:
    """pr_kernel.py and rev_kernel.py (or a second copy of pr_kernel.py) as separate modules,
    in IMPLS order. The first one owns the shared workspace."""
    files = {"pr": "pr_kernel.py", "rev": "pr_kernel.py" if aa else "rev_kernel.py"}
    return {name: load_module(f"{name}_kernel", Path(__file__).parent / files[name]) for name in IMPLS}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def share_workspace(kernels: dict, vocabs: set[int]) -> None:
    """Allocate the first kernel's workspace for the largest batch per vocab (so it never grows),
    and point the second at it."""
    device = torch.device("cuda", torch.cuda.current_device())
    first, second = kernels.values()
    for vocab in vocabs:
        first.buffers(device, max(BATCHES), vocab)
    second._BUFFERS, second._SYNC = first._BUFFERS, first._SYNC


def suite_configs(bench, suite: str, qwen: torch.Tensor) -> list:
    if suite == "gate":
        return gate_configs(bench, qwen)
    return bench.create_benchmark_configs(BATCHES, [131072, 248320]) + bench.create_real_configs(BATCHES, qwen.shape[1])


def run_case(bench, config, rows: torch.Tensor, args: argparse.Namespace, meta: dict) -> None:
    logits = bench.tile_rows(rows, config.batch_size) if config.real_logits else bench.create_logits(config.batch_size, config.vocab_size)
    meta |= dict(case=config.name, batch=config.batch_size, vocab=config.vocab_size)
    append(args.out / "check.jsonl", [meta | check(bench, logits, config)])
    for i in range(args.passes):
        order = IMPLS if i % 2 == 0 else IMPLS[::-1]
        append(args.out / "times.jsonl", [meta | time_impl(bench, name, logits, config, i) for name in order])
    del logits
    torch.cuda.empty_cache()


def check(bench, logits: torch.Tensor, config) -> dict:
    """rev's output equals pr's bit for bit, and both match the exact reference."""
    k, p = config.k_values, config.p_values
    out = {name: bench.IMPLEMENTATIONS[name](logits.clone(), k, p) for name in IMPLS}
    checks = bench.check_implementations(logits, config, IMPLS)
    same = torch.equal(out["pr"].isinf(), out["rev"].isinf()) and torch.equal(out["pr"], out["rev"])
    return dict(kind="check", equal=same, **{f"{n}_close": c["close"] for n, c in checks.items()},
                **{f"{n}_max_tv": c["max_total_variation"] for n, c in checks.items()})  # fmt: skip


def time_impl(bench, name: str, logits: torch.Tensor, config, i: int) -> dict:
    times = bench.bench_gpu_time(bench.IMPLEMENTATIONS[name], logits, config.k_values, config.p_values, 5, 20)
    times = sorted(1000 * t for t in times)
    return dict(impl=name, pass_=i, us=sum(times) / len(times), us_med=times[len(times) // 2])


def append(path: Path, rows: list[dict]) -> None:
    with path.open("a") as file:
        file.writelines(json.dumps(r) + "\n" for r in rows)


if __name__ == "__main__":
    main()
