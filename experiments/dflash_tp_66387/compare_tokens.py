"""Compare two dumped token streams from run_acceptance.py --dump-tokens."""

import argparse
import json


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def first_divergence(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    args = parser.parse_args()

    for ra, rb in zip(load(args.a), load(args.b)):
        same_prompt = ra["prompt_token_ids"] == rb["prompt_token_ids"]
        tok_a, tok_b = ra["token_ids"], rb["token_ids"]
        diverge = first_divergence(tok_a, tok_b)
        match = diverge / max(len(tok_a), len(tok_b), 1)
        print(
            f"req {ra['i']:2d} prompt_same={same_prompt} "
            f"len={len(tok_a)}/{len(tok_b)} first_diff={diverge} "
            f"prefix_match={match:.3f}"
        )


if __name__ == "__main__":
    main()
