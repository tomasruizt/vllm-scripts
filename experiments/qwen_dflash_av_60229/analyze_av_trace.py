"""Summarize an av-trace.jsonl: profiled cost curve, AV budget decisions and per-step output.

Steps are grouped by concurrency level using their request count (c32 cells run ~32 requests, c128 ~128).
"""

import argparse
import json
from pathlib import Path

import numpy as np

LEVELS = {32: (28, 40), 128: (115, 140)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    records = [json.loads(line) for line in parser.parse_args().trace.read_text().splitlines()]
    by_kind = {k: [r for r in records if r["kind"] == k] for k in ("costs", "budget", "output")}
    for r in by_kind["costs"]:
        print_curve(r)
    for c, (lo, hi) in LEVELS.items():
        print_budgets(c, [r for r in by_kind["budget"] if lo <= r["num_reqs"] <= hi])
        print_outputs(c, [r for r in by_kind["output"] if lo <= r["num_reqs"] <= hi], by_kind["output"])


def print_curve(r: dict) -> None:
    print(f"cudagraph_limit={r['cudagraph_limit']}")
    print("verify curve (tokens: ms):", ", ".join(f"{n}: {ms:.2f}" for n, ms in r["verify_curve"]))
    print("draft curve (reqs: ms):", ", ".join(f"{n}: {ms:.2f}" for n, ms in r["draft_curve"][:12]), "...")


def print_budgets(c: int, rows: list[dict]) -> None:
    for prefill in (False, True):
        sel = [r for r in rows if r["has_prefill"] == prefill]
        if not sel:
            continue
        frac = np.array([r["budget"] / r["max_budget"] for r in sel if r["max_budget"]])
        tokens = np.array([r["non_draft_total"] + r["budget"] for r in sel])
        print(f"c{c} {'mixed' if prefill else 'decode'}: steps={len(sel)} admitted/max drafts "
              f"mean={frac.mean():.3f} p10={np.quantile(frac, .1):.3f} p50={np.median(frac):.3f} "
              f"| step tokens p50={np.median(tokens):.0f} | predicted cost p50={np.median([r['cost_ms'] for r in sel]):.2f} ms "
              f"(at max {np.median([r['cost_ms_max'] for r in sel]):.2f})")


def print_outputs(c: int, rows: list[dict], all_rows: list[dict]) -> None:
    """Step time = gap to the previous output record; tokens = tokens sampled in the step."""
    times = {id(r): i for i, r in enumerate(all_rows)}
    gaps, tokens = [], []
    for r in rows:
        i = times[id(r)]
        if i:
            gaps.append((r["time_ns"] - all_rows[i - 1]["time_ns"]) / 1e6)
            tokens.append(r["num_sampled"])
    gaps, tokens = np.array(gaps), np.array(tokens)
    fast = gaps <= 60
    print(f"c{c} output: steps={len(gaps)} fast={fast.mean():.2f} | fast step ms mean={gaps[fast].mean():.1f} "
          f"tokens/step={tokens[fast].mean():.1f} -> {gaps[fast].sum() / tokens[fast].sum() * c:.2f} ms/token/req "
          f"| slow step ms mean={gaps[~fast].mean():.1f} tokens/step={tokens[~fast].mean():.1f} "
          f"| overall ms/token/req={gaps.sum() / tokens.sum() * c:.2f}")


if __name__ == "__main__":
    main()
