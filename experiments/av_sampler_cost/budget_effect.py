"""Phase 3, offline: how much does a per-row price for the untimed part of sample() change the AV budget?

Replays `get_num_tokens` on the verify-all traces of speculators#1430 (temp 0) with and without
the term `price * (num_reqs + b)`, for three base cost models (see ../av_shared_bar). Both budgets
are scored with the realized acceptance and with the cost that includes the term, so the gain is
the modeled throughput gain if the base model plus the term were the true step cost.

Usage: budget_effect.py DATA_RELEASE PRICES_PARQUET   (writes results/budget_effect.parquet)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "av_shared_bar"))
from shared_bar import COST_MODELS, Curves, fit_wall, load_trace, steady, step_curves  # noqa: E402

RUNS = {"Qwen3-4B": "av-alloc-4b-mixed-cg-20261120", "Qwen3.8-27B": "av-alloc-27b-mixed-cg-20261120"}


def main(data: Path, prices_path: Path) -> None:
    prices = pd.read_parquet(prices_path)
    prices = prices[prices["variant"] == "main"]
    prices = prices.assign(untimed_us=prices["head_us"] + prices["sampler_us"])
    # One price per model and mode: the cell with the most rows has the best fit.
    prices = prices.sort_values("max_rows").groupby(["model", "mode"]).tail(1)
    rows = [row for model, run in RUNS.items() for row in replay_run(data / "traces" / run, prices[prices["model"] == model])]
    pd.DataFrame(rows).to_parquet(HERE / "results/budget_effect.parquet")


def replay_run(run_dir: Path, prices: pd.DataFrame) -> list[dict]:
    traces = {m: load_trace(run_dir, m) for m in ("policy", "verify_all")}
    wall, trace = fit_wall(traces), traces["verify_all"]
    greedy_us = float(prices.loc[prices["mode"] == "greedy", "untimed_us"].iloc[0])
    # The wall fit is on temp 0 traces, so its slot price already holds the greedy part.
    in_base = {cm: greedy_us if cm == "wall" else 0.0 for cm in COST_MODELS}
    rows = []
    for (_, c), workload in trace.steps.groupby(["dataset", "concurrency"], sort=True):
        curves = [step_curves(s, trace, wall) for s in steady(workload).itertuples()]
        rows += [dict(model=p.model, mode=p.mode, c=c, cost_model=cm, price_us=p.untimed_us - in_base[cm],
                      **effect(curves, cm, (p.untimed_us - in_base[cm]) * 1e-3))
                 for p in prices.itertuples() for cm in COST_MODELS]
    return rows


def effect(curves: list[Curves], cost_model: str, price_ms: float) -> dict:
    stock = [budget(c, c.cost(cost_model)) for c in curves]
    fixed = [budget(c, true_cost(c, cost_model, price_ms)) for c in curves]
    scheduled = sum(len(c.G) - 1 for c in curves)
    return dict(steps=len(curves), verified_stock=sum(stock) / scheduled, verified_fixed=sum(fixed) / scheduled,
                gain=throughput(curves, fixed, cost_model, price_ms) / throughput(curves, stock, cost_model, price_ms) - 1)


def budget(c: Curves, cost: np.ndarray) -> int:
    return int(np.argmax((c.n + c.G) / cost))


def throughput(curves: list[Curves], budgets: list[int], cost_model: str, price_ms: float) -> float:
    tokens = sum(c.n + c.R[b] for c, b in zip(curves, budgets))
    return tokens / sum(true_cost(c, cost_model, price_ms)[b] for c, b in zip(curves, budgets))


def true_cost(c: Curves, cost_model: str, price_ms: float) -> np.ndarray:
    return c.cost(cost_model) + price_ms * (c.num_reqs + np.arange(len(c.G)))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
