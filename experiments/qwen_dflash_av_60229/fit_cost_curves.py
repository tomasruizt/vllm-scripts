"""Fit end-to-end decode costs from controlled traces, export dense AV curves."""

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import subprocess

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-shape", choices=("concave", "monotone"), default="concave")
    parser.add_argument("--holdout-drafts", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = trace_steps(args.source)
    write_parquet(rows, args.output / "steps.parquet")
    groups = defaultdict(list)
    for row in rows:
        if ((row["mode"].startswith("off") or row["mode"].startswith("oncap")) and row["decode"] and row["usable"]
                and row["cell"] == "main" and row["num_reqs"] == row["concurrency"]):
            groups[(row["num_reqs"], row["num_tokens"])].append(row["step_ms"])
    # Exact full-concurrency shapes avoid mixing the drain with steady decode.
    points = [(r, t, float(np.median(ms)), len(ms)) for (r, t), ms in sorted(groups.items())
              if r in (1, 8, 32, 64, 128) and len(ms) >= 20]
    curves = fit(points, args.token_shape)
    modes = sorted({row["mode"] for row in rows})
    if modes and all(mode.startswith("oncap") for mode in modes):
        curves["fit"].update(calibration_modes=modes, configured_drafter_k=7, token_shape=args.token_shape)
    (args.output / "cost-curves.json").write_text(json.dumps(curves, indent=2) + "\n")
    if args.holdout_drafts is not None:
        validation = validate_shapes(points, args.holdout_drafts)
        (args.output / "fit-validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    comparison = compare_startup(points, curves, args.source)
    if comparison:
        with (args.output / "profile-comparison.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)
    draw_fit(points, curves, args.output, comparison)
    print(json.dumps(curves["fit"], indent=2))


def validate_shapes(points: list[tuple[int, int, float, int]], drafts: int) -> dict:
    heldout = [p for p in points if p[1] == (drafts + 1) * p[0]]
    training = [p for p in points if p not in heldout]
    result = dict(heldout_drafts=drafts, training_points=len(training), heldout_points=heldout, models={})
    for shape in ("concave", "monotone"):
        model = fit(training, shape, min_shapes=2)
        predicted = [model["draft_curve"][r][1] + model["verify_curve"][t][1] for r, t, _, _ in heldout]
        errors = np.array(predicted) - [p[2] for p in heldout]
        result["models"][shape] = dict(predicted_ms=predicted, residual_ms=errors.tolist(),
                                       rmse_ms=float(np.sqrt(np.mean(errors ** 2))))
    return result


def fit(points: list[tuple[int, int, float, int]], token_shape: str = "concave", min_shapes: int = 3) -> dict:
    reqs = sorted({p[0] for p in points})
    if reqs != [1, 8, 32, 64, 128]:
        raise ValueError(f"Missing calibration request counts: {reqs}")
    local = []
    for r in reqs:
        samples = [p for p in points if p[0] == r]
        if len(samples) < min_shapes:
            raise ValueError(f"Need >={min_shapes} token shapes at R={r}")
        x = np.array([[1, p[1]] for p in samples])
        intercept, slope = np.linalg.lstsq(x, [p[2] for p in samples], rcond=None)[0]
        local.append(dict(num_reqs=r, intercept_ms=float(intercept), per_token_ms=float(slope)))
    # Non-decreasing token cost plus a separate intercept at each R.
    # The optional concave basis constrains marginal slopes to decrease.
    knots = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024], dtype=float)
    tokens = np.array([p[1] for p in points])
    request_basis = np.array([[float(p[0] >= r) for r in reqs] for p in points])
    design = np.column_stack([request_basis, token_basis(tokens, knots, token_shape)])
    target = np.array([p[2] for p in points])
    coef = nonnegative_fit(design, target)
    intercepts = np.cumsum(coef[:len(reqs)])
    residual = design @ coef - target
    draft = np.interp(np.arange(513), reqs, intercepts)
    # Beyond measured R, use the last non-negative secant rather than a flat tail.
    draft[129:] += (np.arange(129, 513) - 128) * (intercepts[-1] - intercepts[-2]) / 64
    sizes = np.arange(1025)
    verify = token_basis(sizes, knots, token_shape) @ coef[len(reqs):]
    return dict(draft_curve=list(enumerate(draft.tolist())), verify_curve=list(enumerate(verify.tolist())),
                fit=dict(local_fits=local, token_knots=knots.tolist(), token_coefficients_ms=coef[len(reqs):].tolist(),
                         intercepts_ms=intercepts.tolist(), rmse_ms=float(np.sqrt(np.mean(residual ** 2))),
                         max_abs_error_ms=float(np.max(np.abs(residual))), points=points,
                         point_columns=["num_reqs", "num_tokens", "median_ms", "steps"],
                         method=f"equal-weight shape medians, monotone request intercepts, {token_shape} token cost, nonnegative least squares",
                         timing="step start to next step start, includes target, drafter and CPU overhead",
                         tail="above 1024 tokens retain startup verify-cost increments"))


def token_basis(tokens: np.ndarray, knots: np.ndarray, shape: str) -> np.ndarray:
    if shape == "concave":
        return np.minimum(tokens[:, None], knots) / knots
    starts = np.concatenate(([0.0], knots[:-1]))
    return np.clip((tokens[:, None] - starts) / (knots - starts), 0, 1)


def nonnegative_fit(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    coef = np.zeros(design.shape[1])
    residual = target.copy()
    for _ in range(20000):
        old = coef.copy()
        for j in range(len(coef)):
            column = design[:, j]
            residual += column * coef[j]
            coef[j] = max(0.0, float(column @ residual / (column @ column)))
            residual -= column * coef[j]
        if np.max(np.abs(coef - old)) < 1e-9:
            break
    return coef


def write_parquet(rows: list[dict], output: Path) -> None:
    staging = output.with_suffix(".jsonl")
    staging.write_text("".join(json.dumps(row) + "\n" for row in rows))
    source_sql = str(staging).replace("'", "''")
    output_sql = str(output).replace("'", "''")
    query = f"COPY (SELECT * FROM read_json_auto('{source_sql}', maximum_object_size=268435456)) TO '{output_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    subprocess.run(["duckdb", "-c", query], check=True)
    staging.unlink()


def compare_startup(points: list[tuple[int, int, float, int]], curves: dict, source: Path) -> list[dict]:
    records = []
    for path in source.glob("on*/repeat-*/av-trace.jsonl"):
        records.extend(json.loads(line) for line in path.read_text().splitlines() if '"kind":"costs"' in line)
    if not records:
        return []
    profile = records[-1]
    result = []
    for r, t, measured, n in points:
        if t > profile["cudagraph_limit"]:
            raise ValueError("Startup comparison requires graph-captured decode shapes")
        startup = 0.0
        for key, index in (("draft_curve", r), ("verify_curve", t)):
            sizes, values = np.array(profile[key]).T
            startup += np.maximum.accumulate(values)[min(np.searchsorted(sizes, index), len(sizes) - 1)]
        corrected = curves["draft_curve"][r][1] + curves["verify_curve"][t][1]
        result.append(dict(num_reqs=r, num_tokens=t, steps=n, measured_ms=measured,
                           startup_ms=float(startup), corrected_ms=corrected))
    return result


def draw_fit(points: list[tuple[int, int, float, int]], curves: dict, output: Path, comparison: list[dict] | None = None) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/qwen-av-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    draft = np.array(curves["draft_curve"])[:, 1]
    verify = np.array(curves["verify_curve"])[:, 1]
    for r in (1, 8, 32, 64, 128):
        selected = [p for p in points if p[0] == r]
        tokens = [p[1] for p in selected]
        line, = ax.plot(tokens, draft[r] + verify[tokens], label=f"R={r}")
        ax.scatter(tokens, [p[2] for p in selected], color=line.get_color(), s=22)
        startup = [p["startup_ms"] for p in comparison or [] if p["num_reqs"] == r]
        if startup:
            ax.plot(tokens, startup, color=line.get_color(), linestyle="--", alpha=.65)
    title = "Measured medians (dots), corrected costs (solid), startup costs (dashed)" if comparison else "Measured medians (dots) and corrected cost model (lines)"
    ax.set(xlabel="Executed target tokens", ylabel="Decode step time (ms)", title=title)
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    for ext in ("png", "svg"):
        fig.savefig(output / f"calibrated-costs.{ext}", dpi=160)
    plt.close(fig)


def trace_steps(source: Path) -> list[dict]:
    rows = []
    for path in sorted(source.glob("*/repeat-*/av-trace.jsonl")):
        windows = cell_windows(path.parent)
        records = [json.loads(line) for line in path.read_text().splitlines()]
        steps = [r for r in records if r["kind"] == "step"]
        for current, following in zip(steps, steps[1:]):
            window = next((w for w in windows if w["start"] <= current["time_ns"] < w["end"]), None)
            if window is None:
                continue
            ms = (following["time_ns"] - current["time_ns"]) / 1e6
            rows.append(dict(suite=source.name, mode=path.parts[-3], repeat=int(path.parts[-2].split("-")[-1]),
                             concurrency=window["concurrency"], cell=window["cell"],
                             steady=window["steady_start"] <= current["time_ns"] < window["steady_end"],
                             **current, step_ms=ms, decode=current["prefill_tokens"] == 0,
                             usable=0 < ms < 100 and current["num_reqs"] == following["num_reqs"]
                             and following["time_ns"] < window["end"]))
    return rows


def cell_windows(directory: Path) -> list[dict]:
    windows = []
    for path in sorted(directory.glob("artifacts*/**/profile_export_aiperf.json")):
        data = json.loads(path.read_text())
        phase = next(p for p in data["input_config"]["phases"] if p["name"] == "profiling")
        requests = [json.loads(line)["metadata"] for line in path.with_name("profile_export.jsonl").read_text().splitlines()]
        ends = sorted(r["request_end_ns"] for r in requests)
        windows.append(dict(start=min(r["request_start_ns"] for r in requests),
                            end=max(r["request_end_ns"] for r in requests), concurrency=phase["concurrency"],
                            steady_start=ends[min(phase["concurrency"], len(ends)) - 1],
                            steady_end=max(r["request_start_ns"] for r in requests),
                            cell="sentinel" if path.relative_to(directory).parts[0].endswith("-end") else "main"))
    return windows


if __name__ == "__main__":
    main()
