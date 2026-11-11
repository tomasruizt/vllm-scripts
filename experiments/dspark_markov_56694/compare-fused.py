"""Per-round paired comparison of the fused PR arms (dense / union32 / unfused)."""

import argparse
import itertools
import math
from pathlib import Path

import pandas as pd

PAIRS = [("union32", "dense"), ("union32_unfused", "dense"), ("union32", "union32_unfused")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    frame = pd.read_parquet(args.root / "metrics.parquet")
    frame = frame[frame["valid"]]
    rows = [row(frame, workload, concurrency, left, right)
            for workload, concurrency in frame[["workload", "concurrency"]].drop_duplicates().itertuples(index=False)
            for left, right in PAIRS]
    table = pd.DataFrame(rows)
    table.to_csv(args.root / "paired.csv", index=False)
    print(table.to_string(index=False))


def row(frame: pd.DataFrame, workload: str, concurrency: int, left: str, right: str) -> dict:
    cell = frame[(frame.workload == workload) & (frame.concurrency == concurrency)]
    pivot = cell.pivot_table(index="round", columns="mode", values="throughput")
    ratios = (pivot[left] / pivot[right]).dropna()
    tpot = cell.pivot_table(index="round", columns="mode", values="mean_tpot_ms")
    al = cell.pivot_table(index="round", columns="mode", values="acceptance_length")
    return dict(workload=workload, concurrency=concurrency, left=left, right=right, rounds=len(ratios),
                left_tok_s=round(pivot[left].median(), 1), right_tok_s=round(pivot[right].median(), 1),
                ratio=round(ratios.median(), 4), ratio_min=round(ratios.min(), 4), ratio_max=round(ratios.max(), 4),
                positive=int((ratios > 1).sum()), p_signflip=round(signflip(ratios), 4),
                tpot_change_pct=round(100 * (tpot[left].median() / tpot[right].median() - 1), 2),
                al_left=round(al[left].median(), 3), al_right=round(al[right].median(), 3))


def signflip(ratios: pd.Series) -> float:
    """Exact two-sided sign-flip test on the mean log ratio."""
    values = [math.log(value) for value in ratios]
    observed = abs(sum(values) / len(values))
    hits = sum(abs(sum(sign * value for sign, value in zip(signs, values)) / len(values)) >= observed - 1e-12
               for signs in itertools.product((1, -1), repeat=len(values)))
    return hits / 2 ** len(values)


if __name__ == "__main__":
    main()
