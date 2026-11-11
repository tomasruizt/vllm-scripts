"""AIME avg@n against an OpenAI-compatible server, with the server's default sampling
(the model's generation_config) and thinking on.

Usage: aime_eval.py --port PORT --data aime25.jsonl --n 8 --max-tokens 36000 --out aime.json
The data file has one {"problem", "answer"} object per line (math-ai/aime25).
"""

import argparse
import json
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

PROMPT = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def main() -> None:
    args = parse_args()
    problems = [json.loads(line) for line in open(args.data)]
    client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="x", timeout=7200)
    start = time.time()
    with ThreadPoolExecutor(len(problems)) as pool:
        rows = list(pool.map(lambda p: solve(client, p, args), problems))
    write_summary(rows, args, time.time() - start)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=36000)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def solve(client: OpenAI, problem: dict, args: argparse.Namespace) -> dict:
    """n samples of one problem, graded by the last \\boxed{} answer."""
    response = client.chat.completions.create(
        model="served",
        messages=[{"role": "user", "content": PROMPT.format(problem=problem["problem"])}],
        n=args.n,
        max_tokens=args.max_tokens,
    )
    answers = [boxed(choice.message.content or "") for choice in response.choices]
    correct = [same(answer, problem["answer"]) for answer in answers]
    finish = [choice.finish_reason for choice in response.choices]
    return dict(id=problem.get("id"), answer=problem["answer"], answers=answers, correct=correct, finish=finish, completion_tokens=response.usage.completion_tokens)


def boxed(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return matches[-1].strip() if matches else None


def same(answer: str | None, target) -> bool:
    if answer is None:
        return False
    digits = re.sub(r"[^0-9\-]", "", answer)
    return digits != "" and digits.lstrip("0") == str(target).strip().lstrip("0")


def write_summary(rows: list[dict], args: argparse.Namespace, seconds: float) -> None:
    per_repeat = [statistics.mean(row["correct"][i] for row in rows) for i in range(args.n)]
    summary = dict(
        avg_at_n=statistics.mean(per_repeat),
        repeat_std=statistics.stdev(per_repeat) if args.n > 1 else 0.0,
        n=args.n,
        problems=len(rows),
        truncated=sum(f == "length" for row in rows for f in row["finish"]),
        completion_tokens=sum(row["completion_tokens"] for row in rows),
        seconds=seconds,
    )
    json.dump(dict(summary=summary, rows=rows), open(args.out, "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
