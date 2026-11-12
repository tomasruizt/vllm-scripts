"""Markdown tables of the #66792 description from a fixed-timing parquet (gpu, suite, scenario,
batch, vocab, impl in {triton, new}, us). Prints the Claims rows, then per GPU the real-row
tables and the synthetic geomean table, each under a "### gpu/table" marker.

Usage: pr_tables_queue.py results/fixed-timing-queue/results.parquet
"""

import math
import sys

import duckdb

PEAK_TBS = {"b300": 8.0, "b200": 8.0, "h100": 3.35, "a100": 2.04, "rtx": 1.79, "mi355x": 8.0}
NAMES = {"b300": "B300", "b200": "B200", "h100": "H100", "a100": "A100", "rtx": "RTX PRO 6000", "mi355x": "MI355X (ROCm)"}
SYNTH = [("topk_whole", "top-k 50"), ("topk_topp_whole", "top-k 100, top-p 0.9"),
         ("topk_partial", "half top-k 50, half no-op"), ("mixed_partial", "thirds: top-k 50, top-p 0.5, both"),
         ("topp_whole", "top-p 0.9 (nuclei of 15k to 23k tokens)"), ("topp_partial", "half top-p 0.9, half no-op")]
RANGES = [("8–16", 8, 16), ("64–128", 64, 128), ("256–2048", 256, 2048)]
REAL = [("qwen-topk-topp", "report", "real_topk_topp", 7), ("glm-topp", "glm", "real_topp", 5), ("qwen-topp", "report", "real_topp", 7)]


def main() -> None:
    rows = duckdb.sql(f"pivot (from '{sys.argv[1]}') on impl using first(us) group by gpu, suite, scenario, batch, vocab").fetchall()
    data = {(g, su, sc, b, v): (t, n) for g, su, sc, b, v, n, t in rows}
    gpus = [g for g in PEAK_TBS if any(k[0] == g for k in data)]
    print("### claims")
    for g in gpus:
        print(claims_row(g, {k: x for k, x in data.items() if k[0] == g and k[1] == "report"}))
    for g in gpus:
        for name, suite, scenario, k in REAL:
            print(f"### {g}/{name}")
            print(real_table(g, data, suite, scenario, k))
        print(f"### {g}/synthetic")
        print(synthetic_table(g, data))


def fmt(x: float, digits: int = 1, sign: bool = True) -> str:
    if round(x, digits) == 0:
        x = 0.0
    s = f"{x:+.{digits}f}" if sign and x != 0 else f"{x:.{digits}f}"
    return s.replace("-", "−")


def claims_row(g: str, cases: dict) -> str:
    ratios = [n / t for t, n in cases.values()]
    deltas = [n - t for t, n in cases.values()]
    faster = sum(r <= 0.9 for r in ratios)
    slower = sum(r >= 1.1 for r in ratios)
    geo = 100 * (math.exp(sum(map(math.log, ratios)) / len(ratios)) - 1)
    n = len(ratios)
    return f"| {NAMES[g]} | {faster} / {n} | {slower} / {n} | {fmt(geo, 0, False)}% | {fmt(min(deltas), 1, False)} | {fmt(max(deltas))} |"


def real_table(g: str, data: dict, suite: str, scenario: str, k: int) -> str:
    lines = [f"| rows | reqs@K={k} | SOL µs | Triton µs | New kernel µs | delta µs | % | Triton SOL % | New kernel SOL % |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (gg, su, sc, b, v), (t, n) in sorted(data.items(), key=lambda kv: kv[0][3]):
        if (gg, su, sc) != (g, suite, scenario) or t is None or n is None:
            continue
        sol = 2 * b * v * 4 / (PEAK_TBS[g] * 1e6)
        reqs = b / (k + 1)
        reqs_s = f"{reqs:.0f}" if reqs == int(reqs) else f"{reqs:.1f}"
        lines.append(f"| {b} | {reqs_s} | {sol:.1f} | {t:.1f} | {n:.1f} | {fmt(n - t)} | {fmt(100 * (n / t - 1), 0)}% | {100 * sol / t:.0f}% | {100 * sol / n:.0f}% |")
    return "\n".join(lines)


def synthetic_table(g: str, data: dict) -> str:
    """Per scenario and batch range: geomeans of the kernel times, their difference, and
    the geomean change (the ratio of the two geomeans)."""
    lines = ["| scenario | rows | Triton µs | New kernel µs | delta µs | change |", "|---|---|---:|---:|---:|---:|"]
    for scenario, label in SYNTH:
        for i, (rng, lo, hi) in enumerate(RANGES):
            pairs = [(t, n) for (gg, su, sc, b, v), (t, n) in data.items()
                     if (gg, su, sc) == (g, "report", scenario) and lo <= b <= hi and t and n]
            triton, new = (geomean([p[j] for p in pairs]) for j in (0, 1))
            change = 100 * (new / triton - 1)
            lines.append(f"| {label if i == 0 else ''} | {rng} | {triton:.1f} | {new:.1f} | {fmt(new - triton)} | {fmt(change, 0)}% |")
    return "\n".join(lines)


def geomean(xs: list[float]) -> float:
    return math.exp(sum(map(math.log, xs)) / len(xs))


if __name__ == "__main__":
    main()
