"""Gather every run's replay into one long-format parquet, plot the headroom, print tables.

Inputs: results/<run>/replay.parquet and residuals.parquet (from replay.py).
Outputs: results/all_runs.parquet, results/all_residuals.parquet, plots/headroom.png.
"""

import sys
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reporting.plots import save_figure, scatter_series  # noqa: E402

HERE = Path(__file__).parent
# run id -> metadata. `exclude` lists concurrencies whose traces are confounded.
RUNS = {
    "av-alloc-headroom-20261120": dict(model="Qwen3-4B", tp=1, workload="single-dataset", thinking=True,
                                       full_graphs=False, exclude={128: "verify-all step above the 1,024-token graph cap"}),
    "av-alloc-4b-mixed-20261120": dict(model="Qwen3-4B", tp=1, workload="mixed", thinking=False,
                                       full_graphs=False, exclude={}),
    "av-alloc-27b-mixed-20261120": dict(model="Qwen3.8-27B", tp=1, workload="mixed", thinking=False,
                                        full_graphs=False, exclude={}),
    "av-alloc-4b-mixed-cg-20261120": dict(model="Qwen3-4B", tp=1, workload="mixed", thinking=True, full_graphs=True,
                                          exclude={256: "verify-all censored by the logits cap",
                                                   512: "verify-all censored by the logits cap"}),
    "av-alloc-27b-mixed-cg-20261120": dict(model="Qwen3.8-27B", tp=1, workload="mixed", thinking=True,
                                           full_graphs=True, exclude={}),
    "av-alloc-4b-mixed-nocap-20261120": dict(model="Qwen3-4B", tp=1, workload="mixed", thinking=True,
                                             full_graphs=True, exclude={}),
    "av-alloc-27b-tp2-nocap-20261120": dict(model="Qwen3.8-27B", tp=2, workload="mixed", thinking=True,
                                            full_graphs=True, exclude={}),
}
COLORS = {"Qwen3-4B": "#2a78d6", "Qwen3.8-27B": "#eb6834"}


def main() -> None:
    steps = pd.concat([load_run(run, meta, "replay.parquet") for run, meta in RUNS.items()])
    residuals = pd.concat([load_run(run, meta, "residuals.parquet") for run, meta in RUNS.items()])
    steps.to_parquet(HERE / "results/all_runs.parquet")
    residuals.to_parquet(HERE / "results/all_residuals.parquet")
    gains = gains_table()
    plot_headroom(gains[gains["valid"]])
    print(gains[gains["valid"]].to_string(index=False))
    print(residual_table().to_string(index=False))


def load_run(run: str, meta: dict, name: str) -> pd.DataFrame:
    df = pd.read_parquet(HERE / "results" / run / name)
    df["run"], df["model"], df["tp"] = run, meta["model"], meta["tp"]
    df["thinking"], df["full_graphs"] = meta["thinking"], meta["full_graphs"]
    df["exclusion"] = df["concurrency"].map(lambda c: meta["exclude"].get(c, ""))
    return df


def gains_table() -> pd.DataFrame:
    """Tokens-per-step gain of each policy over greedy, per run, workload, c and budget."""
    return duckdb.sql(f"""
        with g as (
          select run, model, tp, thinking, dataset, concurrency c, budget_frac budget, exclusion, policy,
                 sum(tokens) t, sum(num_reqs) n, count(*) steps,
                 sum(budget) filter (policy = 'greedy') b, sum(max_budget) filter (policy = 'greedy') m
          from '{HERE}/results/all_runs.parquet' group by all),
        base as (select run, dataset, c, budget, t bt, n, steps, b, m from g where policy = 'greedy')
        select run, model, tp, thinking, dataset, c, budget, exclusion = '' valid,
               base.steps, round(base.n / base.steps, 0) mean_reqs, round(base.b / base.m, 3) verified_frac,
               {", ".join(f"round(100 * (max(t) filter (policy = '{p}') / max(bt) - 1), 2) \"{p}\""
                          for p in ("oracle", "hindsight", "calib", "calib_fast", "ucb_1.0", "thompson", "verify_all"))}
        from g join base using (run, dataset, c, budget)
        group by all order by model, run, dataset, c, budget""").df()


def residual_table() -> pd.DataFrame:
    """Per-request persistence of the estimator's error, on the uncensored labels."""
    return duckdb.sql(f"""
        with s as (
          select run, model, dataset, concurrency c, req_id, step, sum_y - sum_p res, n_obs,
                 row_number() over (partition by run, dataset, concurrency, req_id order by step) i
          from '{HERE}/results/all_residuals.parquet' where n_obs > 0 and exclusion = ''),
        halves as (
          select run, model, dataset, c, req_id,
                 sum(res) filter (i % 2 = 0) / sum(n_obs) filter (i % 2 = 0) e,
                 sum(res) filter (i % 2 = 1) / sum(n_obs) filter (i % 2 = 1) o
          from s group by all having count(*) >= 20),
        lagged as (
          select run, dataset, c, res, lag(res) over (partition by run, dataset, c, req_id order by step) prev from s)
        select h.run, h.model, h.dataset, h.c, count(*) reqs, round(corr(e, o), 2) split_half_corr,
               round(stddev(e), 3) sd_request_residual,
               (select round(corr(res, prev), 3) from lagged l
                where l.run = h.run and l.dataset = h.dataset and l.c = h.c) lag1_corr
        from halves h group by all order by h.model, h.run, h.dataset, h.c""").df()


def plot_headroom(gains: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for model, color in COLORS.items():
        rows = gains[gains["model"] == model]
        own = rows[rows["budget"].isin(["policy", "chunk_cap"])].to_dict("records")
        forced = rows[~rows["budget"].isin(["policy", "chunk_cap"])].to_dict("records")
        scatter_series(ax, own, "verified_frac", "oracle", s=36, color=color, label=f"{model}: perfect allocation, AV's budget")
        scatter_series(ax, forced, "verified_frac", "oracle", s=36, facecolors="none", edgecolors=color,
                       label=f"{model}: perfect allocation, forced budget")
    scatter_series(ax, gains.to_dict("records"), "verified_frac", "hindsight", s=14, marker="x", color="#6b6a66",
                   label="Best per-request correction (both models)")
    ax.axhline(0, color="#b0aea6", linewidth=0.8)
    ax.set_xlabel("Share of draft tokens that get verified")
    ax.set_ylabel("Extra tokens per step vs. AV today (%)")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, HERE / "plots/headroom")


if __name__ == "__main__":
    main()
