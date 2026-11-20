"""Does a richer (still lossless) calibration of the same q_max feature beat greedy?

Fits a per-position binned calibration of logit(p) on the lower-concurrency workloads of a
verify-all run, then replays greedy with the recalibrated p on the highest concurrency
(out of sample), at AV's budget and at forced fractions.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from replay import allocate, load_steps, logit

NUM_BINS = 20


def main(run_dir: Path) -> None:
    steps = load_steps(run_dir / "verify_all")
    c_test = steps["concurrency"].max()
    table = fit_table(steps[steps["concurrency"] < c_test])
    test = steps[steps["concurrency"] == c_test]
    rows = [evaluate(test, table, frac) for frac in (None, 0.25, 0.5, 0.75)]
    print(pd.DataFrame(rows).round(2).to_string(index=False))


def evaluate(test: pd.DataFrame, table, frac: float | None) -> dict:
    tokens = {"greedy": 0, "recalib": 0, "oracle": 0}
    for step in test.itertuples():
        a, s = np.asarray(step.accepted), np.asarray(step.scheduled)
        x = logit(np.asarray(step.p))
        budget = min(step.policy_budget, int(s.sum())) if frac is None else round(frac * s.sum())
        tokens["greedy"] += np.minimum(allocate(x, s, budget), a).sum()
        tokens["recalib"] += np.minimum(allocate(apply_table(table, x), s, budget), a).sum()
        tokens["oracle"] += min(budget, a.sum())
    n = sum(len(step.accepted) for step in test.itertuples())
    base = n + tokens["greedy"]
    return dict(budget="policy" if frac is None else frac,
                recalib_pct=100 * ((n + tokens["recalib"]) / base - 1),
                oracle_pct=100 * ((n + tokens["oracle"]) / base - 1))


def fit_table(steps: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per position: quantile bin edges of logit(p) and the empirical logit acceptance."""
    xs, ys = labels_by_position(steps)
    table = []
    for x, y in zip(xs, ys):
        edges = np.quantile(x, np.linspace(0, 1, NUM_BINS + 1)[1:-1])
        idx = np.searchsorted(edges, x)
        rate = (np.bincount(idx, y, NUM_BINS) + 1) / (np.bincount(idx, minlength=NUM_BINS) + 2)
        table.append((edges, logit(np.maximum.accumulate(rate))))
    return table


def labels_by_position(steps: pd.DataFrame) -> tuple[list[np.ndarray], list[np.ndarray]]:
    k = len(steps.iloc[0]["p"][0])
    xs, ys = [[] for _ in range(k)], [[] for _ in range(k)]
    for step in steps.itertuples():
        x = logit(np.asarray(step.p))
        for i, a in enumerate(step.accepted):
            for pos in range(min(step.scheduled[i], a + 1)):
                xs[pos].append(x[i, pos])
                ys[pos].append(pos < a)
    return [np.array(x) for x in xs], [np.array(y, dtype=float) for y in ys]


def apply_table(table, x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for pos, (edges, values) in enumerate(table):
        out[:, pos] = values[np.searchsorted(edges, x[:, pos])]
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]))
