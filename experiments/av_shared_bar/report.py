"""Tables for the shared-bar study, from the parquets that shared_bar.py writes.

Writes `results/report.md`. Gains are throughput (realized tokens / cost) of a policy over the
per-step ratio rule on the same steps, with a block-bootstrap 95% interval over steps.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "results"
KEYS = ["run", "dataset", "concurrency", "window", "cost_model"]
POLICIES = {"bar_x1.0": "bar_fp", "bar_ema10": "ema10", "bar_ema100": "ema100", "bar_ema1000": "ema1000"}
BLOCK, DRAWS = 25, 400
CAP_BINDING_MAX = 0.01  # workloads where the 1 GB logits cap binds more often are excluded


def main() -> None:
    steps = pd.read_parquet(OUT / "steps.parquet")
    replay = usable(pd.read_parquet(OUT / "replay.parquet"), steps)
    steps = steps.merge(replay[KEYS].drop_duplicates(), on=KEYS)
    sections = {
        "Phase 0: offline budget == logged policy_budget": fidelity_table(pd.read_parquet(OUT / "fidelity.parquet")),
        "Wall-clock cost fit (ms)": pd.read_parquet(OUT / "wall_fit.parquet").round(4),
        "Phase 1: r_t / lam, steady window": ratio_table(steps[steps["window"] == "steady"]),
        "Phase 2: gain over the per-step ratio in %, steady window": gain_table(replay[replay["window"] == "steady"]),
        "Phase 2: same, whole workload (includes ramp-up and ramp-down)": gain_table(replay[replay["window"] == "all"]),
        "Phase 2: summary over workloads (gain in %)": summary_table(replay),
        "Breakdown: share of drafts verified, steady window": breakdown_table(replay[replay["window"] == "steady"], steps),
    }
    text = "\n\n".join(f"## {title}\n\n{df.to_markdown(index=False)}" for title, df in sections.items())
    (OUT / "report.md").write_text(text + "\n")
    print(text)


def usable(replay: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    cap = steps.groupby(KEYS)["cap_binding"].mean().rename("cap_share").reset_index()
    return replay.merge(cap[cap["cap_share"] <= CAP_BINDING_MAX][KEYS], on=KEYS)


def fidelity_table(f: pd.DataFrame) -> pd.DataFrame:
    f = f.assign(trimmed=f["logged"] < f["max_budget"])
    g = f.groupby(["run", "mode"]).agg(steps=("same", "size"), same=("same", "mean"), trimmed=("trimmed", "mean"))
    return g.round(4).reset_index()


def ratio_table(steps: pd.DataFrame) -> pd.DataFrame:
    def one(g: pd.DataFrame) -> pd.Series:
        x = g["r"] / g["lam"]
        q = x.quantile([0.05, 0.5, 0.95]).round(2).tolist()
        return pd.Series(dict(steps=len(g), prefill_share=g["prefill"].mean().round(2), lam_tok_ms=g["lam"].iloc[0].round(2),
                              cv=(g["r"].std() / g["r"].mean()).round(3), mean_prefill=x[g["prefill"]].mean().round(2),
                              mean_decode=x[~g["prefill"]].mean().round(2), p5=q[0], p50=q[1], p95=q[2]))

    return steps.groupby(KEYS).apply(one, include_groups=False).reset_index().drop(columns="window")


def gain_table(replay: pd.DataFrame) -> pd.DataFrame:
    rows = [dict(zip(KEYS, key)) | workload_gains(g) for key, g in replay.groupby(KEYS)]
    return pd.DataFrame(rows).drop(columns="window")


def workload_gains(g: pd.DataFrame) -> dict:
    wide = g.pivot(index="step", columns="policy", values=["tokens", "cost_ms", "wall_ms", "budget"])
    out = dict(steps=len(wide), verified=share(wide, "ratio"), verified_bar=share(wide, "bar_x1.0"))
    for policy, label in POLICIES.items():
        gain, lo, hi = bootstrap_gain(wide, policy, "cost_ms")
        out[label] = f"{gain:+.2f} [{lo:+.2f}, {hi:+.2f}]" if label in ("bar_fp", "ema100") else round(gain, 2)
    out["best_bar"] = round(max(gain_pct(wide, p, "cost_ms") for p in wide["tokens"].columns if p.startswith("bar_x")), 2)
    out["bar_fp_wall"] = round(gain_pct(wide, "bar_x1.0", "wall_ms"), 2)
    out["ema100_wall"] = round(gain_pct(wide, "bar_ema100", "wall_ms"), 2)
    return out


def share(wide: pd.DataFrame, policy: str) -> float:
    return round(wide["budget"][policy].sum() / wide["budget"]["verify_all"].sum(), 3)


def gain_pct(wide: pd.DataFrame, policy: str, cost: str, idx: np.ndarray | slice = slice(None)) -> float:
    tput = lambda p: wide["tokens"][p].to_numpy()[idx].sum() / wide[cost][p].to_numpy()[idx].sum()  # noqa: E731
    return 100 * (tput(policy) / tput("ratio") - 1)


def bootstrap_gain(wide: pd.DataFrame, policy: str, cost: str) -> tuple[float, float, float]:
    rng = np.random.default_rng(0)
    n_blocks = max(1, len(wide) // BLOCK)
    starts = rng.integers(0, max(1, len(wide) - BLOCK + 1), size=(DRAWS, n_blocks))
    draws = [gain_pct(wide, policy, cost, (s[:, None] + np.arange(BLOCK)).ravel() % len(wide)) for s in starts]
    return gain_pct(wide, policy, cost), *np.percentile(draws, [2.5, 97.5])


def summary_table(replay: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (window, model), g in replay.groupby(["window", "cost_model"]):
        gains = pd.DataFrame([{p: gain_pct(w.pivot(index="step", columns="policy", values=["tokens", "cost_ms"]), p, "cost_ms")
                               for p in POLICIES} for _, w in g.groupby(KEYS)]).rename(columns=POLICIES)
        for label in gains:
            rows.append(dict(window=window, cost_model=model, policy=label, workloads=len(gains), min=gains[label].min(),
                             median=gains[label].median(), max=gains[label].max()))
    return pd.DataFrame(rows).round(2)


def breakdown_table(replay: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    """Verified share of the scheduled drafts under each rule, split by step type."""
    flags = steps[KEYS + ["step", "prefill", "above_graph"]]
    wide = replay[replay["policy"].isin(["ratio", "bar_x1.0", "verify_all"])].pivot(
        index=KEYS + ["step"], columns="policy", values=["budget", "cost_ms"]).reset_index()
    wide.columns = ["_".join(c).rstrip("_") for c in wide.columns]
    df = wide.merge(flags, on=KEYS + ["step"])
    g = df.groupby(["run", "concurrency", "cost_model", "prefill", "above_graph"])
    sums = g[["budget_ratio", "budget_bar_x1.0", "budget_verify_all", "cost_ms_ratio"]].sum()
    total_cost = sums.groupby(["run", "concurrency", "cost_model"])["cost_ms_ratio"].transform("sum")
    out = pd.DataFrame(dict(steps=g.size(), cost_share=sums["cost_ms_ratio"] / total_cost,
                            verified_ratio=sums["budget_ratio"] / sums["budget_verify_all"],
                            verified_bar=sums["budget_bar_x1.0"] / sums["budget_verify_all"]))
    return out.round(3).reset_index()


if __name__ == "__main__":
    main()
