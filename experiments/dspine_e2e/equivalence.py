"""Greedy GSM8K outputs, acceptance length and accuracy of the DSpine draft.

Run once per build (same GPU), then compare the JSON files: a refactor that
must not change the computation must produce identical token ids.
"""

import argparse
import json
import os
import re

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from datasets import load_dataset  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402

TARGET, TARGET_REV = "Qwen/Qwen3.8-27B", "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
DRAFT = "tencent/Qwen3.8-27B-DSpine-b16"


def main() -> None:
    args = parse_args()
    rows = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n))
    llm = make_llm(args.draft_revision)
    prompts = [[{"role": "user", "content": r["question"]}] for r in rows]
    params = SamplingParams(temperature=0.0, max_tokens=512)
    outputs = llm.chat(prompts, params, chat_template_kwargs={"enable_thinking": False})
    metrics = {m.name: getattr(m, "value", None) for m in llm.get_metrics()}
    drafts = metrics["vllm:spec_decode_num_drafts"]
    accepted = metrics["vllm:spec_decode_num_accepted_tokens"]
    result = dict(
        acceptance_len=1 + accepted / drafts,
        accuracy=sum(map(is_correct, outputs, rows)) / len(rows),
        token_ids=[list(o.outputs[0].token_ids) for o in outputs],
    )
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"acceptance_len={result['acceptance_len']:.3f} accuracy={result['accuracy']:.3f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--draft-revision", required=True)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--out", required=True)
    return p.parse_args()


def make_llm(draft_revision: str) -> LLM:
    return LLM(
        model=TARGET,
        revision=TARGET_REV,
        dtype="bfloat16",
        language_model_only=True,
        max_model_len=4096,
        max_num_seqs=32,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=False,
        generation_config="vllm",
        seed=0,
        disable_log_stats=False,
        attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 3},
        speculative_config={
            "model": DRAFT,
            "revision": draft_revision,
            "method": "dflash",
            "num_speculative_tokens": 15,
        },
    )


def is_correct(output, row) -> bool:
    expected = row["answer"].split("####")[-1].strip().replace(",", "")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", output.outputs[0].text.replace(",", ""))
    return bool(numbers) and float(numbers[-1]) == float(expected)


if __name__ == "__main__":
    main()
