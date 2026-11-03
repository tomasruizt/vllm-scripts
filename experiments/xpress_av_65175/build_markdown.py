"""Markdown summary of the steady-state sweep for the PR description."""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def label(suite: str, commit: str) -> str:
    side = "PR" if suite.endswith("-pr") else "main"
    return f"{side} (`{commit[:10]}`)"


def main() -> None:
    table = pd.read_parquet(HERE / "results/perf.parquet")
    table = table[(table.cell == "main") & table.valid]
    blocks = [side_by_side(name, frame) for name, frame in table.groupby("suite", sort=False)]
    text = "\n\n".join(blocks)
    out = HERE / "report/perf/summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(text)


def side_by_side(suite: str, frame: pd.DataFrame) -> str:
    medians = frame.groupby(["concurrency", "mode"]).agg(
        throughput=("throughput", "median"), tpot=("mean_tpot_ms", "median"),
        al=("acceptance_length", "median"), runs=("repeat", "count")).reset_index()
    lines = [f"### {label(suite, frame.commit.iloc[0])}", "",
             "| c | AV off tok/s | AV on tok/s | Δ | AV off TPOT ms | AV on TPOT ms | AL off | AL on | runs off/on |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c, cell in medians.groupby("concurrency"):
        off, on = cell.set_index("mode").loc["off"], cell.set_index("mode").loc["on"]
        delta = f"{100 * (on.throughput / off.throughput - 1):+.1f}%"
        lines.append(f"| {c} | {off.throughput:,.0f} | {on.throughput:,.0f} | {delta} | "
                     f"{off.tpot:.2f} | {on.tpot:.2f} | {off.al:.2f} | {on.al:.2f} | "
                     f"{off.runs:g}/{on.runs:g} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
