"""Gather the step traces of all fetched runs into one long-format parquet.

Usage: analyze.py RESULTS_DIR   (writes RESULTS_DIR/steps.parquet and cost_tables.parquet)
"""

import json
import sys
from pathlib import Path

import pandas as pd


def main(results: Path) -> None:
    runs = sorted(p.parent for p in results.glob("*/serve-command.txt"))
    steps = pd.concat([load_steps(run) for run in runs], ignore_index=True)
    steps.to_parquet(results / "steps.parquet")
    pd.concat([load_cost_tables(run) for run in runs], ignore_index=True).to_parquet(
        results / "cost_tables.parquet"
    )
    pd.concat([load_aiperf(run) for run in runs], ignore_index=True).to_parquet(
        results / "aiperf.parquet"
    )
    print(steps.groupby(["run", "model"]).size())


def load_aiperf(run: Path) -> pd.DataFrame:
    """One row per AIPerf client: aiperf/<phase>-<budget mode>-<mode>-c<c>-g<clients>."""
    rows = [aiperf_row(p) for p in sorted(run.glob("aiperf/*/profile_export_aiperf.json"))]
    return pd.DataFrame(rows).assign(run=run.name, model=served_model(run), variant=variant(run))


def aiperf_row(path: Path) -> dict:
    phase, budget_mode, mode, c, clients = path.parent.name.split("-")
    summary = json.loads(path.read_text())
    return {
        "phase": phase, "budget_mode": budget_mode, "mode": mode, "c": int(c[1:]),
        "clients": int(clients[1:]),
        "output_tps": summary["output_token_throughput"]["avg"],
        "itl_ms": summary["inter_token_latency"]["avg"],
        "ttft_ms": summary["time_to_first_token"]["avg"],
        "requests": summary["request_count"]["avg"],
        "output_tokens": summary["total_output_tokens"]["avg"],
        "duration_s": summary["benchmark_duration"]["avg"],
    }


def load_steps(run: Path) -> pd.DataFrame:
    frames = [pd.read_json(f, lines=True) for f in sorted(run.glob("steps.jsonl.[0-9]*"))]
    steps = pd.concat([f for f in frames if len(f)], ignore_index=True)
    steps = steps[steps["tag"] != ""].copy()  # steps before the first cell
    phase_mode_c = steps["tag"].str.split(" ", expand=True)
    steps["phase"], steps["mode"] = phase_mode_c[0], phase_mode_c[1]
    steps["c"] = phase_mode_c[2].str.removeprefix("c").astype(int)
    steps["sampler_ms"] = steps["sample_ms"] - steps["head_ms"]
    return steps.assign(run=run.name, model=served_model(run), variant=variant(run))


def load_cost_tables(run: Path) -> pd.DataFrame:
    tables = json.loads(next(run.glob("steps.jsonl.cost_tables.json")).read_text())
    rows = [
        {"table": name, "size": size, "cost_ms": cost}
        for name, costs in tables.items()
        for size, cost in enumerate(costs)
    ]
    return pd.DataFrame(rows).assign(run=run.name, model=served_model(run), variant=variant(run))


def variant(run: Path) -> str:
    """The vLLM tree of the run: main, or main with vllm#62960 merged."""
    return "pr62960" if "pr62960" in run.name else "main"


def served_model(run: Path) -> str:
    return (run / "serve-command.txt").read_text().split()[2].split("/")[-1]


if __name__ == "__main__":
    main(Path(sys.argv[1]))
