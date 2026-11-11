"""End-to-end check of the kept-token sampling path: generate with seeded top-k/top-p
sampling, with or without ngram spec decode, and record the token ids plus how often
the kept-token kernels ran. Run once per arm and compare the outputs.

Usage, from the vllm tree: python kept_tokens_e2e.py OUT.json [--dense] [--spec]
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # Patches below must reach the worker.

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.v1.worker.gpu.sample import sampler as sampler_module  # noqa: E402
from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rs_module  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
CALLS: Counter = Counter()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--dense", action="store_true", help="disable the kept-token path")
    parser.add_argument("--spec", action="store_true", help="ngram spec decode, K=7")
    args = parser.parse_args()
    patch_kernels(args.dense)
    spec = None
    if args.spec:
        spec = dict(method="ngram", num_speculative_tokens=7, prompt_lookup_max=4)
    llm = LLM(
        MODEL,
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        enable_prefix_caching=False,
        speculative_config=spec,
    )
    prompts = make_prompts()
    params = [
        SamplingParams(temperature=0.8, top_k=20, top_p=0.95, max_tokens=256, seed=i)
        for i in range(len(prompts))
    ]
    outputs = llm.generate(prompts, params)
    record = dict(
        dense=args.dense,
        spec=args.spec,
        calls=dict(CALLS),
        token_ids=[list(o.outputs[0].token_ids) for o in outputs],
    )
    args.output.write_text(json.dumps(record))
    print(json.dumps({k: v for k, v in record.items() if k != "token_ids"}))


def patch_kernels(dense: bool) -> None:
    if dense:
        sampler_module.Sampler._max_kept_tokens = lambda self, *a, **k: None
    kept_fn = sampler_module.top_k_top_p_kept_tokens_triton
    sampler_module.top_k_top_p_kept_tokens_triton = counted("sample", kept_fn)
    rs_module.top_k_top_p_kept_tokens_triton = counted("verify", kept_fn)


def counted(name: str, fn):
    def wrapper(*args, **kwargs):
        CALLS[name] += 1
        return fn(*args, **kwargs)

    return wrapper


def make_prompts() -> list[str]:
    topics = ["a binary search", "a linked list", "quicksort", "a hash map"]
    styles = ["in Python", "in C", "in Rust", "in Go"]
    return [f"Write {t} {s}, then repeat the code once more." for t in topics for s in styles]


if __name__ == "__main__":
    main()
