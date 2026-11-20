"""Write AIPerf single_turn prompt files from RedHatAI/speculator_benchmarks subsets.

`mixed` is every subset interleaved in a seeded shuffle, so each batch mixes code, math,
writing, QA, RAG, summarization, translation and tool calls.
"""

import json
import random
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

ALL_SUBSETS = ["HumanEval", "math_reasoning", "writing", "qa", "question", "rag",
               "summarization", "translation", "tool_call"]


def main(out_dir: Path, subsets: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for subset in subsets:
        prompts = mixed() if subset == "mixed" else load(subset)
        lines = [json.dumps({"text": prompt}) for prompt in prompts]
        (out_dir / f"{subset}.jsonl").write_text("\n".join(lines) + "\n")
        print(subset, len(prompts))


def mixed() -> list[str]:
    prompts = [p for subset in ALL_SUBSETS for p in load(subset)]
    random.Random(0).shuffle(prompts)
    return prompts


def load(subset: str) -> list[str]:
    src = hf_hub_download("RedHatAI/speculator_benchmarks", f"{subset}.jsonl", repo_type="dataset")
    return [json.loads(line)["prompt"] for line in open(src)]


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
