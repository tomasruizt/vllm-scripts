"""Write AIPerf single_turn prompt files from RedHatAI/speculator_benchmarks subsets."""

import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


def convert(subset: str, out_dir: Path) -> None:
    src = hf_hub_download("RedHatAI/speculator_benchmarks", f"{subset}.jsonl", repo_type="dataset")
    prompts = [json.loads(line)["prompt"] for line in open(src)]
    lines = [json.dumps({"text": prompt}) for prompt in prompts]
    (out_dir / f"{subset}.jsonl").write_text("\n".join(lines) + "\n")
    print(subset, len(prompts))


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    for subset in sys.argv[2:]:
        convert(subset, out_dir)
