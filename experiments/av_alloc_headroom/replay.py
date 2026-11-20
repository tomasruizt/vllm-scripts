"""Replay AV draft allocation policies on a verify-all trace.

Under AV_TRACE_VERIFY_ALL=1 every drafted token is verified, so each trace row holds the
uncensored accepted length `a` of a request. At the per-step budget B that the real AV
policy chose (`policy_budget`), we replay:

- greedy:   production ranking, top-B slots by survival (cumprod of p).
- oracle:   knows `a`, accepts min(B, sum a). Upper bound for any within-step reallocation.
- calib:    greedy on p shifted by a per-request logit offset u_r, learned online only from
            the labels the policy itself verified (censored, like in production).
- ucb_L:    calib plus L * sd(u_r), so rarely-verified requests get explored.
- thompson: calib with u_r sampled from its posterior.

Writes `<run>/replay.parquet` (long format, one row per step and policy) and
`<run>/residuals.parquet` (one row per request and step, for the persistence check).
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PRIOR_PRECISION = 4.0  # tau: prior precision of the per-request logit offset
UCB_LAMBDAS = (0.5, 1.0, 2.0)
EPS = 1e-6
# None replays at the policy's own budget, a float forces B = frac * scheduled drafts.
BUDGET_FRACS = (None, 0.25, 0.5, 0.75)


def main(run_dir: Path) -> None:
    print("greedy replay == production widths:", check_fidelity(load_steps(run_dir / "policy")))
    steps = load_steps(run_dir / "verify_all")
    workloads = [w for _, w in steps.groupby(["dataset", "concurrency"])]
    results = [replay_workload(w, frac) for frac in BUDGET_FRACS for w in workloads]
    out = pd.concat([r[0] for r in results])
    residuals = pd.concat([r[1] for r in results if r[0]["budget_frac"].iloc[0] == "policy"])
    out.to_parquet(run_dir / "replay.parquet")
    residuals.to_parquet(run_dir / "residuals.parquet")
    print(summary(out).to_string())


def summary(out: pd.DataFrame) -> pd.DataFrame:
    out = out[out["budget_frac"] == "policy"]
    g = out.groupby(["dataset", "concurrency", "policy"])[["tokens", "accepted"]].sum()
    base = g.xs("greedy", level="policy")["tokens"]
    g["gain_pct"] = 100 * (g["tokens"] / base.reindex(g.index.droplevel("policy")).values - 1)
    return g["gain_pct"].unstack("policy").round(2)


def check_fidelity(steps: pd.DataFrame) -> str:
    """On the production (policy) trace, replayed greedy widths must equal admitted."""
    same = [
        np.array_equal(allocate(logit(np.asarray(s.p)), np.asarray(s.scheduled),
                                min(s.policy_budget, sum(s.scheduled))), s.admitted)
        for s in steps.itertuples()
    ]
    return f"{np.mean(same):.4f} of {len(same)} steps"


# ---------------------------------------------------------------- trace loading


def load_steps(mode_dir: Path) -> pd.DataFrame:
    manifest = [json.loads(line) for line in open(mode_dir / "manifest.jsonl")]
    records = (json.loads(line) for line in open(mode_dir / "trace.jsonl"))
    steps = [r for r in records if r["kind"] == "step" and r["req_ids"]]
    df = pd.DataFrame(steps)
    df["dataset"], df["concurrency"] = None, 0
    for w in manifest:
        inside = (df["t"] >= w["t0"]) & (df["t"] <= w["t1"])
        df.loc[inside, ["dataset", "concurrency"]] = w["dataset"], w["concurrency"]
    return df[df["dataset"].notna()].sort_values("t").reset_index(drop=True)


# ---------------------------------------------------------------- replay


@dataclass
class RequestBelief:
    """Per-request logit offset u with precision h, fitted by online Newton steps."""

    decay: float = 1.0  # < 1 forgets old labels, so u can follow phases within a request
    u: dict[str, float] = field(default_factory=dict)
    h: dict[str, float] = field(default_factory=dict)

    def get(self, req_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        u = np.array([self.u.get(r, 0.0) for r in req_ids])
        h = np.array([self.h.get(r, 0.0) for r in req_ids])
        return u, h

    def update(self, req_ids, logits, widths, accepted) -> None:
        """Fold in the labels a policy observed: positions < width, up to the first reject."""
        for i, r in enumerate(req_ids):
            n_obs = min(widths[i], accepted[i] + 1)
            if n_obs == 0:
                continue
            y = np.arange(n_obs) < accepted[i]
            p = sigmoid(logits[i, :n_obs] + self.u.get(r, 0.0))
            h = self.decay * self.h.get(r, 0.0) + float((p * (1 - p)).sum())
            self.h[r] = h
            self.u[r] = self.u.get(r, 0.0) + float((y - p).sum()) / (h + PRIOR_PRECISION)


def replay_workload(w: pd.DataFrame, frac: float | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    policies = ["calib", "calib_fast", "thompson"] + [f"ucb_{lam}" for lam in UCB_LAMBDAS]
    beliefs = {name: RequestBelief(decay=0.8 if name == "calib_fast" else 1.0) for name in policies}
    hindsight = hindsight_offsets(w)
    rows, residual_rows = [], []
    for step in w.itertuples():
        a = np.asarray(step.accepted)
        s = np.asarray(step.scheduled)
        logits = logit(np.asarray(step.p))
        budget = min(step.policy_budget, int(s.sum())) if frac is None else round(frac * s.sum())
        u_star = np.array([hindsight[r] for r in step.req_ids])
        widths = {"greedy": allocate(logits, s, budget), "verify_all": s,
                  "hindsight": allocate(logits + u_star[:, None], s, budget)}
        for name in policies:
            u, h = beliefs[name].get(step.req_ids)
            sd = 1 / np.sqrt(h + PRIOR_PRECISION)
            shift = policy_shift(name, u, sd, rng)
            widths[name] = allocate(logits + shift[:, None], s, budget)
            beliefs[name].update(step.req_ids, logits, widths[name], a)
        base = dict(dataset=step.dataset, concurrency=step.concurrency, step=step.Index,
                    budget_frac="policy" if frac is None else str(frac),
                    num_reqs=len(a), budget=budget, max_budget=int(s.sum()))
        for name, width in widths.items():
            acc = int(np.minimum(width, a).sum())
            rows.append(dict(base, policy=name, accepted=acc, tokens=len(a) + acc))
        oracle = min(budget, int(a.sum()))
        rows.append(dict(base, policy="oracle", accepted=oracle, tokens=len(a) + oracle))
        residual_rows += residuals(step, logits, a, s)
    return pd.DataFrame(rows), pd.DataFrame(residual_rows)


def hindsight_offsets(w: pd.DataFrame, iters: int = 8) -> dict[str, float]:
    """Non-causal upper bound for per-request calibration: each request's logit offset
    fitted (Newton) on all of its uncensored labels in the workload."""
    labels: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for step in w.itertuples():
        logits = logit(np.asarray(step.p))
        for i, r in enumerate(step.req_ids):
            n_obs = min(step.scheduled[i], step.accepted[i] + 1)
            y = np.arange(n_obs) < step.accepted[i]
            labels.setdefault(r, []).append((logits[i, :n_obs], y))
    return {r: fit_offset(np.concatenate([x for x, _ in obs]), np.concatenate([y for _, y in obs]), iters)
            for r, obs in labels.items()}


def fit_offset(x: np.ndarray, y: np.ndarray, iters: int) -> float:
    u = 0.0
    for _ in range(iters):
        p = sigmoid(x + u)
        u += float((y - p).sum()) / (float((p * (1 - p)).sum()) + EPS)
    return u


def policy_shift(name: str, u: np.ndarray, sd: np.ndarray, rng) -> np.ndarray:
    if name in ("calib", "calib_fast"):
        return u
    if name == "thompson":
        return u + sd * rng.standard_normal(len(u))
    return u + float(name.removeprefix("ucb_")) * sd


def allocate(logits: np.ndarray, scheduled: np.ndarray, budget: int) -> np.ndarray:
    """Production ranking: top-`budget` slots by survival, as a per-request width."""
    survival = np.cumprod(sigmoid(logits), axis=1)
    eligible = np.arange(logits.shape[1])[None, :] < scheduled[:, None]
    survival = np.where(eligible, survival, -1.0)
    if budget >= int(eligible.sum()):
        return scheduled.copy()
    order = np.argsort(-survival, axis=None, kind="stable")[:budget]
    rows = np.unravel_index(order, survival.shape)[0]
    return np.bincount(rows, minlength=len(scheduled))


def residuals(step, logits: np.ndarray, a: np.ndarray, s: np.ndarray) -> list[dict]:
    """Uncensored per-request residual sum(y - p) over the graded prefix."""
    out = []
    for i, r in enumerate(step.req_ids):
        n_obs = min(s[i], a[i] + 1)
        y = np.arange(n_obs) < a[i]
        p = sigmoid(logits[i, :n_obs])
        out.append(dict(dataset=step.dataset, concurrency=step.concurrency, step=step.Index,
                        req_id=r, n_obs=int(n_obs), sum_y=int(y.sum()), sum_p=float(p.sum()),
                        p0=float(p[0]) if n_obs else np.nan, accepted=int(a[i])))
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
