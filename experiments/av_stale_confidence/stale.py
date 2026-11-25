"""What does the stale budget input cost vLLM adaptive verification (AV)?

AV sizes the step budget B on the host from the confidences of the previous step's drafts
(lag 1, and 1.0 for a request with no landed confidences yet), then ranks slots on the GPU
with the fresh confidences. This replays `get_num_tokens` on the AV traces with different
budget inputs and scores each budget with the realized acceptance of the verify-all traces.

Input: the data release of speculators#1430 (`traces/<run>/<mode>.parquet` + cost tables).
Output: `results/steps.parquet` (one row per step, cost model and budget input) and
`results/requests.parquet` (one row per step and request, stale vs. fresh confidences).
"""

import json
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "av_alloc_headroom"))
from build_report import RUNS  # noqa: E402
from replay import VOCAB, load_mode, serve_model  # noqa: E402

DEFAULT_DATA = HERE.parent / "av_alloc_headroom/results/data-release"
MAX_LAG = 3
WARMUP_STEPS = MAX_LAG  # steps of a workload whose lagged inputs reach into another workload
CHUNK_BYTES = 2**30
# Slot price of the linear cost models, as a multiple of the logged verify table's mean slope.
LINEAR_PRICES = {"linear": 1.0, "linear_cheap": 0.25}
# Mitigation 2: budget from an exponential moving average of the stale sorted survival curve.
EMA_ALPHAS = {"ema_0.3": 0.3, "ema_0.05": 0.05}
EMA_GRID = np.linspace(0, 1, 129)  # the curve is stored at these slot quantiles
# Short outputs put more rows in the 1.0-filled state than these traces have (~1% of rows):
# fill this share of random rows of the fresh input to see what a higher share would cost.
SYNTHETIC_FILL_SHARES = {"fresh_fill_10pct": 0.10, "fresh_fill_25pct": 0.25}


def main(data_dir: Path) -> None:
    traces = sorted((data_dir / "traces").glob("*/*.parquet"))
    results = [analyze_trace(path) for path in traces]
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    pd.concat([r[0] for r in results]).to_parquet(out / "steps.parquet")
    pd.concat([r[1] for r in results]).to_parquet(out / "requests.parquet")


def analyze_trace(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir, mode = path.parent, path.stem
    steps = load_mode(run_dir, mode)
    cost_models = load_cost_models(run_dir, mode, steps)
    prod_max_logits = CHUNK_BYTES // (VOCAB[serve_model(run_dir, mode)] * 4)
    history, rng = History(), np.random.default_rng(0)
    step_rows, request_rows = [], []
    workload_step = steps.groupby(["dataset", "concurrency"]).cumcount()
    for step, rank in zip(steps.itertuples(), workload_step):
        inputs = budget_inputs(step, history, rng)
        base = step_metadata(step, inputs, rank, prod_max_logits)
        step_rows += replay_step(step, inputs, cost_models, base, verify_all=mode == "verify_all")
        request_rows.append(request_drift(step, inputs, base["step"]))
        history.push(step)
    tags = dict(run=run_dir.name, mode=mode, exclusion="")
    step_df, request_df = pd.DataFrame(step_rows).assign(**tags), pd.concat(request_rows).assign(**tags)
    for df in (step_df, request_df):
        df["exclusion"] = df["concurrency"].map(lambda c: RUNS[run_dir.name]["exclude"].get(c, ""))
    print(f"{run_dir.name}/{mode}: {len(steps)} steps")
    return step_df, request_df


# ---------------------------------------------------------------- budget inputs


@dataclass
class History:
    """Confidences of the last steps, by request id."""

    past: deque = field(default_factory=lambda: deque(maxlen=MAX_LAG))
    seen: set[str] = field(default_factory=set)
    ema_curves: dict[str, np.ndarray] = field(default_factory=dict)

    def smoothed_scores(self, stale_scores: np.ndarray) -> dict[str, np.ndarray]:
        """Fold the stale curve into each moving average, return them at this step's slot count."""
        slots = (np.arange(len(stale_scores)) + 0.5) / max(len(stale_scores), 1)
        curve = np.interp(EMA_GRID, slots, stale_scores) if len(stale_scores) else np.zeros_like(EMA_GRID)
        for name, alpha in EMA_ALPHAS.items():
            self.ema_curves[name] = (1 - alpha) * self.ema_curves.get(name, curve) + alpha * curve
        return {name: np.interp(slots, EMA_GRID, ema) for name, ema in self.ema_curves.items()}

    def push(self, step) -> None:
        self.past.append(dict(zip(step.req_ids, np.asarray(step.p))))
        self.seen.update(step.req_ids)

    def lagged(self, req_ids: list[str], num_positions: int, lag: int) -> tuple[np.ndarray, np.ndarray]:
        """The host copy `lag` steps behind the fresh one: 1.0 where the request has no row yet."""
        rows = self.past[-lag] if len(self.past) >= lag else {}
        known = np.array([r in rows for r in req_ids])
        conf = np.ones((len(req_ids), num_positions))
        for i in np.flatnonzero(known):
            conf[i] = rows[req_ids[i]]
        return conf, known


@dataclass
class BudgetInputs:
    confidences: dict[str, np.ndarray]  # variant -> [num_requests, K]
    scores: dict[str, np.ndarray]  # variant -> survival of the valid slots, best first
    new: np.ndarray  # rows filled with 1.0 in the production (stale) input
    gap: np.ndarray  # filled rows of requests that were seen before (row mismatch)


def budget_inputs(step, history: History, rng: np.random.Generator) -> BudgetInputs:
    fresh = np.asarray(step.p)
    stale, known = history.lagged(step.req_ids, fresh.shape[1], lag=1)
    new = ~known[:, None]
    mean_fill = stale[known].mean(axis=0) if known.any() else np.ones(fresh.shape[1])
    confidences = dict(
        stale=stale,  # production
        fresh=fresh,
        fresh_fill=np.where(new, 1.0, fresh),  # only the 1.0 fill is wrong
        drift_only=np.where(new, fresh, stale),  # only the step-to-step drift is wrong
        mean_fill=np.where(new, mean_fill, stale),  # mitigation 1: batch mean per position
        lag2=history.lagged(step.req_ids, fresh.shape[1], lag=2)[0],
        lag3=history.lagged(step.req_ids, fresh.shape[1], lag=3)[0],
    )
    for name, share in SYNTHETIC_FILL_SHARES.items():
        filled = rng.random((len(fresh), 1)) < share
        batch_mean = fresh[~filled[:, 0]].mean(axis=0) if not filled.all() else 1.0
        confidences[name] = np.where(filled, 1.0, fresh)
        confidences[name.replace("fill", "meanfill")] = np.where(filled, batch_mean, fresh)
    gap = ~known & np.array([r in history.seen for r in step.req_ids])
    scheduled = np.asarray(step.scheduled)
    scores = {name: sorted_scores(conf, scheduled) for name, conf in confidences.items()}
    scores.update(history.smoothed_scores(scores["stale"]))
    return BudgetInputs(confidences, scores, ~known, gap)


def sorted_scores(confidences: np.ndarray, scheduled: np.ndarray) -> np.ndarray:
    survival = slot_survival(confidences, scheduled, ineligible=-1.0)
    return np.sort(survival[survival >= 0])[::-1]


# ---------------------------------------------------------------- replay


def replay_step(step, inputs: BudgetInputs, cost_models: dict, base: dict, verify_all: bool) -> list[dict]:
    scheduled = np.asarray(step.scheduled)
    tokens = realized_tokens(step, scheduled) if verify_all else None
    rows = []
    for cost_model, cost_fn in cost_models.items():
        costs = cost_fn(step)
        budgets = {name: choose_budget(scores, step, costs) for name, scores in inputs.scores.items()}
        if cost_model == "table":
            budgets["logged"] = min(step.policy_budget, step.max_budget)
        if verify_all:
            budgets["hindsight"] = int(np.argmax(tokens / costs))
        for name, budget in budgets.items():
            realized = int(tokens[budget]) if verify_all else None
            rows.append(dict(base, cost_model=cost_model, variant=name, budget=budget,
                             tokens=realized, cost_ms=float(costs[budget])))
    return rows


def choose_budget(scores: np.ndarray, step, costs: np.ndarray) -> int:
    """`AdaptiveVerificationManager.get_num_tokens`: argmax of expected tokens per cost."""
    scores = scores[: step.max_budget]
    expected_tokens = step.num_reqs + np.concatenate(([0.0], np.cumsum(scores)))
    return int(np.argmax(expected_tokens / costs))


def realized_tokens(step, scheduled: np.ndarray) -> np.ndarray:
    """Tokens emitted at every budget 0..max_budget, with the production (fresh) ranking."""
    survival = slot_survival(np.asarray(step.p), scheduled, ineligible=-1.0)
    order = np.argsort(-survival, axis=None, kind="stable")[: step.max_budget]
    rows, positions = np.unravel_index(order, survival.shape)
    # Survival falls along a request, so the top slots are a prefix of each request.
    hits = positions < np.asarray(step.accepted)[rows]
    return step.num_reqs + np.concatenate(([0], np.cumsum(hits)))


def slot_survival(confidences: np.ndarray, scheduled: np.ndarray, ineligible: float = 0.0) -> np.ndarray:
    survival = np.cumprod(confidences.astype(np.float32).astype(np.float64), axis=1)
    eligible = np.arange(confidences.shape[1])[None, :] < scheduled[:, None]
    return np.where(eligible, survival, ineligible)


# ---------------------------------------------------------------- cost models


def load_cost_models(run_dir: Path, mode: str, steps: pd.DataFrame) -> dict:
    """`table`: the logged tables (vLLM main: a staircase over the captured CUDA graph sizes).
    `linear*`: the same base cost plus a constant price per draft slot, a proxy for vllm#62960."""
    tables = json.loads((run_dir / f"{mode}_cost_tables.json").read_text())
    draft_ms, verify_ms = np.asarray(tables["draft_ms"]), np.asarray(tables["verify_ms"])
    max_tokens = int((steps["non_draft_total"] + steps["max_budget"]).max())
    mean_slope = (verify_ms[max_tokens] - verify_ms[0]) / max_tokens

    def table(step) -> np.ndarray:
        first = step.non_draft_total
        return draft_ms[step.num_reqs] + verify_ms[first : first + step.max_budget + 1]

    def linear(step, price: float) -> np.ndarray:
        base = draft_ms[step.num_reqs] + verify_ms[step.non_draft_total]
        return base + price * mean_slope * np.arange(step.max_budget + 1)

    models = {name: (lambda step, price=price: linear(step, price)) for name, price in LINEAR_PRICES.items()}
    return dict(table=table, **models)


# ---------------------------------------------------------------- per-step and per-request records


def step_metadata(step, inputs: BudgetInputs, workload_step: int, prod_max_logits: int) -> dict:
    scheduled = np.asarray(step.scheduled)
    expected = {name: float(slot_survival(conf, scheduled).sum()) for name, conf in inputs.confidences.items()}
    sorted_curve = {name: np.sort(slot_survival(inputs.confidences[name], scheduled), axis=None)
                    for name in ("stale", "fresh", "drift_only")}
    drafts = int(scheduled.sum())
    return dict(
        dataset=step.dataset, concurrency=step.concurrency, step=step.Index, warm=workload_step >= WARMUP_STEPS,
        num_reqs=step.num_reqs, num_rows=len(scheduled), num_new=int(inputs.new.sum()), num_gap=int(inputs.gap.sum()),
        drafts=drafts, max_budget=step.max_budget, non_draft_total=step.non_draft_total,
        policy_budget=step.policy_budget, cap_binding=step.max_budget < drafts,
        prod_cap_budget=max(0, prod_max_logits - step.num_reqs),
        expected_fresh=expected["fresh"], expected_stale=expected["stale"], expected_drift_only=expected["drift_only"],
        curve_l1_stale=float(np.abs(sorted_curve["stale"] - sorted_curve["fresh"]).sum()),
        curve_l1_drift_only=float(np.abs(sorted_curve["drift_only"] - sorted_curve["fresh"]).sum()),
    )


def request_drift(step, inputs: BudgetInputs, step_idx: int) -> pd.DataFrame:
    """Stale vs. fresh expected accepted drafts of every request with a landed stale row."""
    scheduled = np.asarray(step.scheduled)
    fresh, stale = inputs.confidences["fresh"], inputs.confidences["stale"]
    known = ~inputs.new
    return pd.DataFrame(dict(
        dataset=step.dataset, concurrency=step.concurrency, step=step_idx,
        expected_fresh=slot_survival(fresh, scheduled).sum(axis=1)[known],
        expected_stale=slot_survival(stale, scheduled).sum(axis=1)[known],
        p0_fresh=fresh[known, 0], p0_stale=stale[known, 0],
        accepted=np.asarray(step.accepted)[known],
    ))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA)
