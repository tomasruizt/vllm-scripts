"""Append collection or matched performance cells to the live AIPerf queue."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collection", "performance", "fill"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--lookup", type=Path)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[256])
    parser.add_argument("--gaps", type=Path)
    parser.add_argument("--fill-round", type=int, default=0)
    args = parser.parse_args()
    if args.stage == "collection":
        base = json.loads(Path(__file__).with_name("cost-c256-collection-plan.json").read_text())[1:]
        plan = [dict(item, phase=item["phase"].replace("c256", f"c{r}"), concurrency=r, requests=2 * r)
                for r in args.concurrencies for item in base]
    elif args.stage == "fill":
        if args.gaps is None:
            parser.error("fill requires --gaps JSON from the coverage audit")
        gaps = json.loads(args.gaps.read_text())
        if isinstance(gaps, list):
            gaps = {str(row["num_reqs"]): row["budgets"] for row in gaps}
        plan = [dict(phase=f"burstfill{args.fill_round}-c{r}", mode="sweep", concurrency=int(r), requests=2 * int(r),
                     fixed_output_tokens=1024, budgets={r: budgets}) for r, budgets in gaps.items() if budgets]
    else:
        if args.lookup is None or not args.lookup.is_file():
            parser.error("performance requires an existing frozen --lookup")
        plan = performance_plan(str(args.lookup), args.reverse)
    append_commands(args.root, plan)


def performance_plan(lookup: str, reverse: bool) -> list[dict]:
    plan = []
    for repeat in range(1, 4):
        modes = ["native", "oracle"] if repeat % 2 else ["oracle", "native"]
        if reverse:
            modes.reverse()
        for mode in modes:
            plan.append(dict(phase=f"r{repeat}-{mode}-c256", mode=mode, concurrency=256,
                             requests=1024, lookup=lookup))
    for mode in (["oracle", "native"] if reverse else ["native", "oracle"]):
        plan.append(dict(phase=f"trace-{mode}-c256-post", mode=mode, concurrency=256,
                         requests=1024, lookup=lookup, trace=True))
    return plan


def append_commands(root: Path, plan: list[dict]) -> None:
    paths = sorted((root / "commands").glob("*.json"))
    phases = {json.loads(path.read_text())["phase"] for path in paths}
    index = max((int(path.stem) for path in paths), default=-1) + 1
    for command in plan:
        if command["phase"] in phases:
            continue
        path = root / "commands" / f"{index:03}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(command))
        temporary.replace(path)
        print(f"queued {index:03} {command['phase']}")
        index += 1


if __name__ == "__main__":
    main()
