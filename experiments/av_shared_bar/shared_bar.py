"""Does AV pick its draft budget with the wrong objective? Offline replay on verify-all traces.

Production picks, per step, B = argmax_b (n + G(b)) / C(b). Long-run throughput is maximized
by one shared bar: B = argmax_b (n + G(b)) - lam * C(b), lam = long-run tokens per ms.
This script replays both rules on the AV allocation traces (data release of
`experiments/av_alloc_headroom`) and writes long-format parquets to `results/`:

- `fidelity.parquet`: does the offline budget equal the logged `policy_budget` (phase 0).
- `steps.parquet`:    per step r_t, the fixed-point lam and flags (phase 1).
- `wall_fit.parquet`: the wall-clock cost fit of each run.
- `replay.parquet`:   per step, cost model and policy: budget, tokens, cost (phase 2).

Usage: shared_bar.py <data-release>/traces [run ...]
"""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "results"
BAR_FACTORS = (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0)  # bar = factor * fixed-point lam
EMA_STEPS = (10, 100, 1000)  # time constants of the causal lam, in steps
STEADY_LOAD = 0.9  # steady window: request count >= this share of its 95th percentile
DEFAULT_CUDAGRAPH_LIMIT = 1024  # Blackwell default when the serve command sets none
COST_MODELS = ("main", "linear", "wall")


def main(traces: Path, runs: list[str]) -> None:
    OUT.mkdir(exist_ok=True)
    run_dirs = [traces / r for r in runs] if runs else sorted(p for p in traces.iterdir() if p.is_dir())
    outputs = [analyze_run(d) for d in run_dirs]
    for i, name in enumerate(("fidelity", "steps", "replay", "wall_fit")):
        pd.concat([o[i] for o in outputs]).to_parquet(OUT / f"{name}.parquet")


def analyze_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    traces = {m: load_trace(run_dir, m) for m in ("policy", "verify_all") if (run_dir / f"{m}.parquet").exists()}
    fidelity = pd.concat([fidelity_rows(t).assign(mode=m) for m, t in traces.items()])
    wall = fit_wall(traces)
    trace = traces["verify_all"]
    workloads = [w for _, w in trace.steps.groupby(["dataset", "concurrency"], sort=True)]
    windows = [(name, select(w)) for w in workloads for name, select in WINDOWS.items()]
    steps, replay = zip(*(replay_workload(w, trace, wall, name) for name, w in windows if len(w) > 1))
    tag = dict(run=run_dir.name)
    print(run_dir.name, f"fidelity {fidelity['same'].mean():.4f}", wall.to_dict("records")[0], flush=True)
    return (fidelity.assign(**tag), pd.concat(steps).assign(**tag), pd.concat(replay).assign(**tag), wall.assign(**tag))


# ---------------------------------------------------------------- phase 2: policies


def steady(w: pd.DataFrame) -> pd.DataFrame:
    """Steps at full load: from the first to the last step with >= 90% of the peak request count.
    Drops the ramp-up and the ramp-down tail, where a low r_t reflects the small batch."""
    at_load = np.flatnonzero(w["num_reqs"] >= STEADY_LOAD * w["num_reqs"].quantile(0.95))
    return w.iloc[at_load[0] : at_load[-1] + 1]


WINDOWS = {"all": lambda w: w, "steady": steady}


def replay_workload(w: pd.DataFrame, trace: "Trace", wall: pd.DataFrame, window: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = [step_curves(s, trace, wall) for s in w.itertuples()]
    keys = dict(dataset=w["dataset"].iloc[0], concurrency=w["concurrency"].iloc[0], window=window)
    step_rows, replay_rows = [], []
    for model in COST_MODELS:
        if model == "wall" and wall["slot_ms"].iloc[0] <= 0:
            continue  # no usable wall-clock slot price for this run
        lam = fixed_point_lam(curves, model)
        budgets = policy_budgets(curves, model, lam)
        replay_rows += [score(c, model, name, b[i]) for name, b in budgets.items() for i, c in enumerate(curves)]
        step_rows += [dict(step=c.step, cost_model=model, lam=lam, r=best_ratio(c, model), prefill=c.prefill,
                           cap_binding=c.cap_binding, num_reqs=c.num_reqs, max_budget=len(c.G) - 1,
                           above_graph=c.above_graph(budgets["ratio"][i])) for i, c in enumerate(curves)]
    return pd.DataFrame(step_rows).assign(**keys), pd.DataFrame(replay_rows).assign(**keys)


def policy_budgets(curves: list["Curves"], model: str, lam: float) -> dict[str, list[int]]:
    realized_lam = fixed_point_lam(curves, model, realized=True)
    budgets = {"ratio": [ratio_budget(c, model) for c in curves],
               "verify_all": [len(c.G) - 1 for c in curves],
               "bar_realized_fp": [bar_budget(c, model, realized_lam) for c in curves]}
    budgets |= {f"bar_x{f}": [bar_budget(c, model, f * lam) for c in curves] for f in BAR_FACTORS}
    budgets |= {f"bar_ema{tau}": ema_budgets(curves, model, tau) for tau in EMA_STEPS}
    return budgets


def ratio_budget(c: "Curves", model: str) -> int:
    """Production: maximize this step's own expected tokens per ms."""
    return int(np.argmax((c.n + c.G) / c.cost(model)))


def bar_budget(c: "Curves", model: str, lam: float) -> int:
    """Shared bar: maximize expected tokens minus lam * cost (Dinkelbach linearization)."""
    return int(np.argmax(c.n + c.G - lam * c.cost(model)))


def fixed_point_lam(curves: list["Curves"], model: str, realized: bool = False, iters: int = 30) -> float:
    """lam = total tokens / total cost under the policy that lam induces."""
    lam = 0.0
    for _ in range(iters):
        b = [bar_budget(c, model, lam) for c in curves]
        tokens = sum(c.n + (c.R if realized else c.G)[bi] for c, bi in zip(curves, b))
        new = tokens / sum(c.cost(model)[bi] for c, bi in zip(curves, b))
        if abs(new - lam) < 1e-9 * new:
            break
        lam = new
    return lam


def ema_budgets(curves: list["Curves"], model: str, tau: int) -> list[int]:
    """Causal lam: ratio of EMAs of realized tokens and cost over past steps only."""
    first = ratio_budget(curves[0], model)
    tokens, cost = curves[0].n + curves[0].R[first], curves[0].cost(model)[first]
    budgets = [first]
    for i, c in enumerate(curves[1:], start=2):
        b = bar_budget(c, model, tokens / cost)
        decay = 1 - max(1 / i, 1 / tau)  # plain running mean until tau steps exist (cold start)
        tokens = decay * tokens + (1 - decay) * (c.n + c.R[b])
        cost = decay * cost + (1 - decay) * c.cost(model)[b]
        budgets.append(b)
    return budgets


def best_ratio(c: "Curves", model: str) -> float:
    return float(np.max((c.n + c.G) / c.cost(model)))


def score(c: "Curves", model: str, policy: str, b: int) -> dict:
    return dict(step=c.step, cost_model=model, policy=policy, budget=b, exp_tokens=c.n + c.G[b],
                tokens=c.n + c.R[b], cost_ms=c.cost(model)[b], wall_ms=c.cost("wall")[b])


# ---------------------------------------------------------------- per-step curves


@dataclass
class Curves:
    """One step, as functions of the draft budget b = 0..max_budget."""

    step: int
    n: int  # tokens the step emits without any draft (sampling requests)
    num_reqs: int
    non_draft: int
    G: np.ndarray  # expected accepted drafts, from the confidences the budget used
    R: np.ndarray  # realized accepted drafts, production ranking on fresh confidences
    costs: dict[str, np.ndarray]
    cudagraph_limit: int
    cap_binding: bool

    def cost(self, model: str) -> np.ndarray:
        return self.costs[model]

    @property
    def prefill(self) -> bool:
        return self.non_draft > self.num_reqs

    def above_graph(self, b: int) -> bool:
        return self.non_draft + b > self.cudagraph_limit


def step_curves(s, trace: "Trace", wall: pd.DataFrame) -> Curves:
    budgets = np.arange(s.max_budget + 1)
    scores = budget_scores(s)
    hits = realized_hits(s)[: s.max_budget]
    w = wall.iloc[0]
    wall_cost = w.const_ms + w.req_ms * s.num_reqs + w.prefill_tok_ms * (s.non_draft_total - s.num_reqs) + w.slot_ms * budgets
    wall_cost = wall_cost if w.slot_ms > 0 else np.full(len(budgets), np.nan)
    costs = dict(main=table_cost(trace.draft_ms, trace.verify_ms, s), wall=wall_cost,
                 linear=table_cost(trace.draft_ms, trace.verify_linear_ms, s))
    return Curves(step=s.Index, n=s.num_reqs, num_reqs=s.num_reqs, non_draft=s.non_draft_total,
                  G=np.concatenate(([0.0], np.cumsum(scores))), R=np.concatenate(([0], np.cumsum(hits))),
                  costs=costs, cudagraph_limit=trace.cudagraph_limit, cap_binding=s.max_budget < sum(s.scheduled))


def budget_scores(s) -> np.ndarray:
    """get_num_tokens: survival from the stale host confidences, sorted, capped at max_budget."""
    return np.sort(valid_survival(s.p_stale, s.scheduled))[::-1][: s.max_budget]


def realized_hits(s) -> np.ndarray:
    """Per slot in production rank order (fresh confidences): was the draft accepted."""
    k = np.arange(len(s.p[0]))
    valid = k[None, :] < np.asarray(s.scheduled)[:, None]
    survival = np.where(valid, np.cumprod(np.asarray(s.p, dtype=np.float64), axis=1), -1.0)
    order = np.argsort(-survival, axis=None, kind="stable")[: int(valid.sum())]
    rows, cols = np.unravel_index(order, survival.shape)
    return cols < np.asarray(s.accepted)[rows]


def valid_survival(p: list, scheduled: list) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    valid = np.arange(p.shape[1])[None, :] < np.asarray(scheduled)[:, None]
    return np.cumprod(p, axis=1)[valid]


def table_cost(draft_ms: np.ndarray, verify_ms: np.ndarray, s) -> np.ndarray:
    return draft_ms[s.num_reqs] + verify_ms[s.non_draft_total : s.non_draft_total + s.max_budget + 1]


# ---------------------------------------------------------------- phase 0: fidelity


def fidelity_rows(trace: "Trace") -> pd.DataFrame:
    """Offline get_num_tokens vs the logged policy_budget, on the logged cost tables."""
    rows = [dict(step=s.Index, concurrency=s.concurrency, dataset=s.dataset, logged=s.policy_budget,
                 offline=offline_budget(s, trace), max_budget=s.max_budget) for s in trace.steps.itertuples()]
    df = pd.DataFrame(rows)
    return df.assign(same=df["logged"] == df["offline"])


def offline_budget(s, trace: "Trace") -> int:
    tokens = s.num_reqs + np.concatenate(([0.0], np.cumsum(budget_scores(s))))
    return int(np.argmax(tokens / table_cost(trace.draft_ms, trace.verify_ms, s)))


# ---------------------------------------------------------------- wall-clock cost fit


def fit_wall(traces: dict[str, "Trace"]) -> pd.DataFrame:
    """Step wall time (gap between trace timestamps) ~ const + requests + prefill tokens + verified
    slots, pooled over the policy and verify-all servers (they verify different budgets in similar
    states, which identifies the slot price), with an offset for the policy server's GPU."""
    frames = [wall_frame(t).assign(policy_server=float(m == "policy")) for m, t in traces.items()]
    df = pd.concat(frames)
    df = df[df["dt_ms"] < df["dt_ms"].quantile(0.99)]
    cols = ["const", "num_reqs", "prefill_tokens", "slots", "policy_server"]
    X = df.assign(const=1.0)[cols].to_numpy(dtype=np.float64)
    beta, *_ = np.linalg.lstsq(X, df["dt_ms"].to_numpy(), rcond=None)
    resid = df["dt_ms"] - X @ beta
    r2 = 1 - (resid**2).sum() / ((df["dt_ms"] - df["dt_ms"].mean()) ** 2).sum()
    return pd.DataFrame([dict(const_ms=beta[0], req_ms=beta[1], prefill_tok_ms=beta[2], slot_ms=beta[3],
                              policy_server_ms=beta[4], r2=r2, n_steps=len(df), n_servers=len(traces))])


def wall_frame(trace: "Trace") -> pd.DataFrame:
    s = trace.steps
    df = pd.DataFrame(dict(dt_ms=1e3 * s.groupby(["dataset", "concurrency"])["t"].diff(), num_reqs=s["num_reqs"],
                           prefill_tokens=s["non_draft_total"] - s["num_reqs"], slots=s["admitted"].map(sum)))
    return df.dropna()


# ---------------------------------------------------------------- trace loading


@dataclass
class Trace:
    steps: pd.DataFrame  # one row per step, per-request columns as lists
    draft_ms: np.ndarray
    verify_ms: np.ndarray
    verify_linear_ms: np.ndarray
    cudagraph_limit: int


def load_trace(run_dir: Path, mode: str) -> Trace:
    rows = pd.read_parquet(run_dir / f"{mode}.parquet")
    per_req = ["req_id", "scheduled", "admitted", "accepted", "p"]
    agg = {c: list if c in per_req else "first" for c in rows.columns.drop("step")}
    steps = rows.groupby("step", sort=True).agg(agg)
    steps["p_stale"] = stale_confidences(steps)
    costs = json.loads((run_dir / f"{mode}_cost_tables.json").read_text())
    verify = np.asarray(costs["verify_ms"])
    limit = re.search(r"max_cudagraph_capture_size\D+(\d+)", (run_dir / f"{mode}_serve_command.txt").read_text())
    return Trace(steps, np.asarray(costs["draft_ms"]), verify, linearize(verify),
                 int(limit.group(1)) if limit else DEFAULT_CUDAGRAPH_LIMIT)


def stale_confidences(steps: pd.DataFrame) -> list[list[np.ndarray]]:
    """The budget reads a host copy that lags the GPU by one step: each request's confidences
    from its previous traced step, 1.0 for a request's first step (the reset value). The trace
    logs only the fresh GPU confidences, this lag is validated by the phase 0 fidelity."""
    last: dict[str, np.ndarray] = {}
    out = []
    for req_ids, p in zip(steps["req_id"], steps["p"]):
        out.append([last.get(r, np.ones(len(row))) for r, row in zip(req_ids, p)])
        last.update(zip(req_ids, p))
    return out


def linearize(verify_ms: np.ndarray) -> np.ndarray:
    """The staircase table with linear interpolation between the captured sizes (bucket right
    edges). Proxy for a cost model with a per-slot price, as vllm#62960 introduces."""
    edges = np.flatnonzero(np.diff(verify_ms) > 1e-9)
    edges = np.concatenate((edges, [len(verify_ms) - 1]))
    return np.interp(np.arange(len(verify_ms)), edges, verify_ms[edges])


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
