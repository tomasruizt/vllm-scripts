"""Compare pinned main/PR rejection-sampling modules with production UVA metadata."""

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import warnings
from collections.abc import Callable
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from cupti import cupti  # noqa: F401, fail rather than time a fallback
from flashinfer.testing import bench_gpu_time_with_cupti
from vllm.v1.worker.gpu.sample.states import SamplingStates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-module", type=Path, required=True)
    parser.add_argument("--pr-module", type=Path, required=True)
    parser.add_argument("--main-commit", required=True)
    parser.add_argument("--pr-commit", required=True)
    parser.add_argument("--requests", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--drafts", type=int, nargs="+", default=[3, 7])
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["dense", "masked"],
        default=["dense", "masked"],
    )
    parser.add_argument("--vocab", type=int, default=248320)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("error", message=".*Falling back.*")
    assert int(version("cupti-python").split(".")[0]) >= 13
    torch.manual_seed(42)
    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": version("triton"),
        "flashinfer": version("flashinfer-python"),
        "cupti": version("cupti-python"),
        "main_commit": args.main_commit,
        "pr_commit": args.pr_commit,
        "main_sha256": digest(args.main_module),
        "pr_sha256": digest(args.pr_module),
        "dtype": "bfloat16",
        "vocab": args.vocab,
        "metadata": "production UVA",
        "timing": "CUPTI, CUDA graph, cold L2",
        "dry_run_ms": 25,
        "measurement_ms": 100,
        "command": sys.argv,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)
    main_sample = load_module(args.main_module, "pinned_main").rejection_sample
    pr_sample = load_module(args.pr_module, "pinned_pr").rejection_sample
    with (args.output / "measurements.jsonl").open("w") as output:
        for scenario in args.scenarios:
            for requests in args.requests:
                for drafts in args.drafts:
                    for accepted in sorted({0, min(3, drafts), drafts}):
                        measure_case(
                            args,
                            main_sample,
                            pr_sample,
                            scenario,
                            requests,
                            drafts,
                            accepted,
                            output,
                        )
    (args.output / "DONE").write_text("ok\n")


def measure_case(
    args: argparse.Namespace,
    main_sample: Callable,
    pr_sample: Callable,
    scenario: str,
    requests: int,
    drafts: int,
    accepted: int,
    output: Any,
) -> None:
    inputs, states = make_inputs(requests, drafts, accepted, args.vocab, scenario)
    arms = {
        "main": partial(main_sample, **inputs),
        "pr": partial(pr_sample, **inputs, all_greedy=True),
    }
    expected, counts = arms["main"]()
    sampled, actual_counts = arms["pr"]()
    assert torch.equal(actual_counts, counts), "Sample counts differ"
    valid = torch.arange(9, device="cuda")[None, :] < counts[:, None]
    assert torch.equal(sampled[valid], expected[valid]), "Sampled token IDs differ"
    assert torch.equal(counts, torch.full_like(counts, accepted + 1))
    assert states.temperature.gpu.device.type == "cuda"
    for repeat in range(args.repeats):
        order = list(arms) if repeat % 2 == 0 else list(reversed(arms))
        for arm in order:
            samples = bench_gpu_time_with_cupti(
                arms[arm],
                use_cuda_graph=True,
                cold_l2_cache=True,
                dry_run_time_ms=25,
                repeat_time_ms=100,
            )
            row = {
                "gpu": torch.cuda.get_device_name(),
                "main_commit": args.main_commit,
                "pr_commit": args.pr_commit,
                "scenario": scenario,
                "requests": requests,
                "target_rows": drafts + 1,
                "accepted": accepted,
                "repeat": repeat,
                "arm": arm,
                "us": statistics.median(samples) * 1000,
                "samples": len(samples),
                "min_us": min(samples) * 1000,
                "max_us": max(samples) * 1000,
                "token_ids_match": True,
                "counts_match": True,
            }
            output.write(json.dumps(row) + "\n")
            output.flush()
            print(json.dumps(row), flush=True)


def make_inputs(
    requests: int, drafts: int, accepted: int, vocab: int, scenario: str
) -> tuple[dict[str, Any], SamplingStates]:
    rows = requests * (drafts + 1)
    logits = torch.randn(rows, vocab, dtype=torch.bfloat16, device="cuda")
    if scenario == "masked":
        values, indices = torch.topk(logits, 20, dim=-1)
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, indices, values)
    tokens = torch.zeros(requests, drafts + 1, dtype=torch.int32, device="cuda")
    tokens[:, 1:] = logits.argmax(-1).view(requests, drafts + 1)[:, :-1]
    if accepted < drafts:
        tokens[:, accepted + 1] = -1
    reqs = torch.arange(requests, dtype=torch.int32, device="cuda")
    states = SamplingStates(requests, vocab)
    states.temperature.np.fill(0.0)
    states.seeds.np.fill(0)
    states.apply_staged_writes()
    inputs = {
        "target_logits": logits,
        "draft_logits_cache": None,
        "draft_sampled": tokens.flatten(),
        "cu_num_logits": torch.arange(requests + 1, dtype=torch.int32, device="cuda")
        * (drafts + 1),
        "pos": torch.arange(rows, device="cuda"),
        "idx_mapping": reqs,
        "expanded_idx_mapping": reqs.repeat_interleave(drafts + 1),
        "expanded_local_pos": torch.arange(drafts + 1, device="cuda").repeat(requests),
        "temperature": states.temperature.gpu,
        "seed": states.seeds.gpu,
        "num_speculative_steps": 8,
        "skip_masked_vocab_blocks": scenario == "masked",
    }
    return inputs, states


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
