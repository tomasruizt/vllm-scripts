"""GPU kernel time of sample() per step and logits row, from vLLM torch-profiler traces.

A kernel belongs to sample() when its launch falls inside an `av_sample` annotation, and to the
LM head when it falls inside `av_lm_head`. Rows per step come from the step trace of the same run.

Usage: prof_sample.py RUN_DIR   (expects traces/<mode>/*.json[.gz]; writes RUN_DIR/sample_kernels.parquet)
"""

import bisect
import gzip
import json
import sys
from pathlib import Path

import pandas as pd

GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
LAUNCH_CATS = ("cuda_runtime", "cuda_driver")  # Triton launches through the driver API


def main(run: Path) -> None:
    rows = profiled_rows_per_step(run)
    frames = [mode_kernels(d).assign(mode=d.name, rows_per_step=rows.get(d.name)) for d in sorted(run.glob("traces/*"))]
    kernels = pd.concat(frames, ignore_index=True)
    kernels["us_per_row"] = kernels["us_per_step"] / kernels["rows_per_step"]
    kernels.to_parquet(run / "sample_kernels.parquet")


def mode_kernels(trace_dir: Path) -> pd.DataFrame:
    """GPU time per step of every kernel launched inside sample(), by phase (LM head or sampler)."""
    events = load_events(trace_dir)
    samples = spans(events, "av_sample")
    heads = spans(events, "av_lm_head")
    launch_ts = {e["args"]["correlation"]: e["ts"] for e in events if e.get("cat") in LAUNCH_CATS and "correlation" in e.get("args", {})}
    gpu = [e for e in events if e.get("cat") in GPU_CATS and e["args"].get("correlation") in launch_ts]
    rows = [dict(phase=phase(launch_ts[e["args"]["correlation"]], samples, heads), kernel=short_name(e["name"]), us=e["dur"]) for e in gpu]
    df = pd.DataFrame(rows)
    df = df[df["phase"] != "other"]
    out = df.groupby(["phase", "kernel"], as_index=False).agg(us=("us", "sum"), calls=("us", "size"))
    return out.assign(steps=len(samples), us_per_step=out["us"] / len(samples), calls_per_step=out["calls"] / len(samples))


def phase(ts: float, samples: list[tuple[float, float]], heads: list[tuple[float, float]]) -> str:
    if inside(ts, heads):
        return "lm_head"
    return "sampler" if inside(ts, samples) else "other"


def inside(ts: float, sorted_spans: list[tuple[float, float]]) -> bool:
    i = bisect.bisect_right(sorted_spans, (ts, float("inf"))) - 1
    return i >= 0 and ts <= sorted_spans[i][1]


def spans(events: list[dict], name: str) -> list[tuple[float, float]]:
    """CPU-side spans of a profiler annotation (the trace repeats them on the GPU timeline)."""
    return sorted((e["ts"], e["ts"] + e["dur"]) for e in events if e.get("name") == name and e.get("cat") == "user_annotation")


def load_events(trace_dir: Path) -> list[dict]:
    trace = max(trace_dir.rglob("*.json*"), key=lambda p: p.stat().st_size)
    opener = gzip.open if trace.suffix == ".gz" else open
    with opener(trace) as f:
        return json.load(f)["traceEvents"]


def short_name(name: str) -> str:
    return name.split("(")[0][:60]


def profiled_rows_per_step(run: Path) -> dict[str, float]:
    """Mean logits rows of the steps traced while the profiler ran, per mode."""
    steps = pd.concat(pd.read_json(f, lines=True) for f in run.glob("steps.jsonl.[0-9]*"))
    profiled = steps[steps["tag"].str.startswith("profiled")]
    return profiled.groupby(profiled["tag"].str.split().str[1])["num_logits"].mean().to_dict()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
