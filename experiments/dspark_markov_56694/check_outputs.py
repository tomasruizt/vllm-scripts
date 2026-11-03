"""Save greedy public-API responses for cross-mode agreement checks."""

import concurrent.futures
import json
import sys
import urllib.request
from pathlib import Path


def main() -> None:
    port = sys.argv[1]
    source = sys.argv[2]
    output = sys.argv[3]
    rows = [json.loads(line) for line in Path(source).read_text().splitlines()[:8]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda row: generate(port, row), rows))
    Path(output).write_text(json.dumps(responses, indent=2) + "\n")


def generate(port: str, row: dict) -> dict:
    payload = dict(model="qwen", messages=[dict(role="user", content=row["turns"][0])],
                   temperature=0, max_tokens=512, seed=42,
                   chat_template_kwargs=dict(enable_thinking=False))
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    return dict(question_id=row["question_id"], **result)


if __name__ == "__main__":
    main()
