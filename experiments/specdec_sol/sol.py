"""Measured spec-decode timelines with semantic roofline references and separate NCU replay diagnostics.

Each GPU op is charged its busy time and preceding idle gap. Unknown work remains unknown, and assumptions,
execution-shape provenance and separate memory/compute bounds accompany supported estimates.
"""

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path

import duckdb
import pandas as pd
import yaml

from kernel_models import classification_name, estimate_record, hardware_profile, kernel_models, query_tokens, shape_models
from benchmarking.nsys_shapes import annotate_shapes
from benchmarking.roofline import Estimate, Work, estimate, price

KERNEL_CATEGORIES = [  # first match wins, matched against the lowercase short name
    ("communication", r"nccl|allreduce|all_reduce|allgather|reduce_scatter|cross_device|custom_ar"),
    ("moe", r"moe|expert|topk_softmax|grouped_gemm|routing|^bmm_"),
    ("gemm", r"nvjet|gemm|cutlass|xmma|cublas|matmul|splitk"),
    ("attention", r"flash|fmha|attn|attention|paged|mla|cascade"),
    ("ssm", r"gdn|delta_rule|mamba|ssm|conv1d|causal_conv"),
    ("norm", r"norm"),
    ("rope", r"rotary|rope"),
    ("activation", r"silu|gelu|act_and_mul|swiglu"),
    ("sampling", r"sampl|topk|top_k|argmax|reject|softmax|prob|gumbel|exponential"),
    ("copy", r"copy|memcpy|memset|cat|index|gather|scatter|fill"),
    ("triton", r"^triton_"),
]
SOL_COLORS = [(1.5, "good"), (3.0, "yellow"), (float("inf"), "terrible")]  # Chrome trace reserved colors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="one concurrency directory, with nsys/ and optionally ncu/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ncu", type=Path, action="append", help="ncu CSV to reuse, repeat for additional profiles from the same GPU type")
    parser.add_argument("--trace-steps", type=int, default=3)
    parser.add_argument("--shape-dtype", help="explicit uniform operand precision assumption for shape estimates, e.g. bf16")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    hw = hardware(args.run_dir / "nsys/trace.sqlite") | {"world_size": world_size(args.run_dir.parent / "config.yaml")}
    db = duckdb.connect()
    db.execute(f"ATTACH '{args.run_dir / 'nsys/trace.sqlite'}' AS t (TYPE sqlite, READ_ONLY)")
    ops = timeline(db)
    ops, annotations = annotate_shapes(db, ops)
    ops = measurement_ops(db, ops, args.run_dir.name)
    ncu_paths = args.ncu or [args.run_dir / "ncu/kernels.csv"]
    previous_meta = args.output / "meta.json"
    shape_dtype = args.shape_dtype
    if shape_dtype is None and previous_meta.exists():
        shape_dtype = json.loads(previous_meta.read_text()).get("shape_dtype_assumption")
    if args.ncu is None and not any(path.exists() for path in ncu_paths) and previous_meta.exists():
        saved = json.loads(previous_meta.read_text())
        sources = saved.get("ncu_sources") or ([saved["ncu_source"]] if saved.get("ncu_source") else [])
        ncu_paths = [Path(source) for source in sources] or ncu_paths
    ops = match_ncu(ops, ncu_profiles(ncu_paths, hw))
    ops = add_costs(ops, hw)
    steps, steady_rule = step_table(db, ops)
    steps, steady_rule = measured_steps(db, steps, steady_rule, args.run_dir.name)
    steps = execution_shapes(db, steps, args.run_dir.parent / "config.yaml")
    info = model_info(args.run_dir / "nsys/model-info.json")
    steps = analytic_sol(steps, ops, info, hw)
    with_layers = info and "layers" in info  # runs from before the probe reported its layer inventory have none
    ops, mapping = kernel_models(ops, steps, info, hw) if with_layers else (ops.assign(model_sol_ns=float("nan"), model_family=None), {})
    ops = shape_models(ops, info or {}, hw, shape_dtype)
    sol_sources(ops, hw)
    steady = steps[steps.steady]
    steady_ops = ops[ops.step.isin(steady.step)]
    ops.to_parquet(args.output / "ops.parquet")
    steps.to_parquet(args.output / "steps.parquet")
    annotations.to_parquet(args.output / "operators.parquet")
    summary = waterfall(steady_ops, steady, hw)
    (args.output / "summary.md").write_text(summary)
    shown = representative_steps(steps, args.trace_steps)
    cpu_ranges(db, ops[ops.step.isin(shown)]).to_parquet(args.output / "ranges.parquet")
    meta = {"hw": hw, "shown_steps": shown, "ncu_sources": [str(path) for path in ncu_paths if path.exists()],
            "shape_dtype_assumption": shape_dtype, "operator_annotations": len(annotations),
            "shape_diagnostic": "annotated measurement window" in steady_rule,
            "checks": checks(db, ops, steady_ops, steps, steady_rule) | mapping}
    (args.output / "meta.json").write_text(json.dumps(meta, indent=2, default=lambda value: value.item()))
    trace = perfetto_trace(db, ops, shown)
    (args.output / "steps.perfetto.json").write_text(json.dumps(trace, default=lambda value: value.item()))  # numpy scalars
    print(summary)


def checks(db: duckdb.DuckDBPyConnection, ops: pd.DataFrame, steady_ops: pd.DataFrame, steps: pd.DataFrame, steady_rule: str) -> dict:
    """What the report rests on, so a run that went wrong is visible in the report itself."""
    busy = steady_ops.busy_ns.sum()
    return {
        "steps": len(steps),
        "steady steps": int(steps.steady.sum()),
        "steady rule": steady_rule,
        "workers in trace": int(db.execute("SELECT count(DISTINCT globalTid) FROM steps").fetchone()[0]),
        "busy time with a SOL estimate": f"{steady_ops.busy_ns[steady_ops.sol_ns.notna()].sum() / busy:.1%}",
        "ops with a CPU phase": f"{(~steady_ops.phase.str.startswith('(')).mean():.1%}",
        "ops with an inferred launch": f"{steady_ops.inferred_launch.mean():.1%}",
        "query shape sources": ", ".join(sorted(steps[steps.steady].query_shape_source.unique())),
        "estimate sources": ", ".join(sorted(steady_ops.sol_source.dropna().unique())),
        "busy time with NSYS input shapes": f"{steady_ops.busy_ns[steady_ops.operator_name.notna()].sum() / busy:.1%}",
        "busy time with NSYS shape estimates": f"{steady_ops.busy_ns[steady_ops.operator_shape_estimate].sum() / busy:.1%}",
        "kernel bounds above measured duration": int(((steady_ops.kind == "kernel") & (steady_ops.sol_ns > steady_ops.dur_ns)).sum()),
        "steady steps with a GPU stall": stall_check(steps[steps.steady], host_gaps(steady_ops)),
    }


# --- Inputs -----------------------------------------------------------------------------------------------------


NVLINK_BYTES_PER_S = 900e9  # NVLink 5 (B200/B300) per direction and GPU, nsys does not record it


def hardware(trace: Path) -> dict:
    """ncu percentages are relative to the same peaks, nsys records the DRAM peak for copies."""
    with sqlite3.connect(trace) as db:
        name, peak_bw = db.execute("SELECT name, max(memoryBandwidth) FROM TARGET_INFO_GPU").fetchone()
    return {"bw": peak_bw, "gpu": name, "link_bytes_per_s": NVLINK_BYTES_PER_S}


def ncu_sol(path: Path, hw: dict) -> pd.DataFrame:
    """Per kernel signature: time at the SOL of its busiest unit, measured DRAM bytes, and the SOL percentage."""
    if not path.exists():
        print(f"no ncu CSV at {path}, only semantic estimates available")
        columns = ["ncu_ns", "sm_pct", "mem_pct", "sol_pct", "kernel_sol_ns", "dram_bytes", "dram_sol_ns"]
        return pd.DataFrame({"signature": pd.Series(dtype=str), "ncu_source": pd.Series(dtype=str),
                             **{c: pd.Series(dtype=float) for c in columns}})
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    rows = [r for r in rows[1:] if r.get("ID", "").strip().isdigit()]  # row 0 holds the units
    devices = {row.get("device__attribute_display_name") for row in rows} - {None, ""}
    if hw.get("gpu") and devices and devices != {hw["gpu"]}:
        raise ValueError(f"NCU GPU mismatch in {path}: {devices}, expected {hw['gpu']}")
    frame = pd.DataFrame(
        {
            "signature": [f"{r['Kernel Name']}|{r['Grid Size']}|{r['Block Size']}" for r in rows],
            "ncu_ns": [number(r["gpu__time_duration.sum"]) for r in rows],
            "sm_pct": [number(r["sm__throughput.avg.pct_of_peak_sustained_elapsed"]) for r in rows],
            "mem_pct": [number(r["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"]) for r in rows],
            "dram_bytes": [number(r["dram__bytes_read.sum"]) + number(r["dram__bytes_write.sum"]) for r in rows],
        }
    )
    frame["sol_pct"] = frame[["sm_pct", "mem_pct"]].max(axis=1)
    frame["kernel_sol_ns"] = frame.ncu_ns * frame.sol_pct / 100
    frame["dram_sol_ns"] = frame.dram_bytes / hw["bw"] * 1e9
    columns = ["ncu_ns", "sm_pct", "mem_pct", "sol_pct", "kernel_sol_ns", "dram_bytes", "dram_sol_ns"]
    return frame.groupby("signature", as_index=False)[columns].median().assign(
        ncu_samples=frame.groupby("signature").size().values, ncu_source=str(path))


def ncu_profiles(paths: list[Path], hw: dict) -> pd.DataFrame:
    """Reuse measured signatures across workloads on the same GPU type, preferring earlier profiles."""
    profiles = [ncu_sol(path, hw).assign(ncu_priority=priority) for priority, path in enumerate(paths)]
    return pd.concat(profiles, ignore_index=True).drop_duplicates("signature", keep="first")


def match_ncu(ops: pd.DataFrame, ncu: pd.DataFrame) -> pd.DataFrame:
    """Exact match on the signature first. Kernels whose grid depends on the data (MoE GEMMs over routed tokens,
    per-step token counts with AV) rarely repeat a grid, those take the median SOL % of the same kernel and block
    size, applied to their own duration (ncu_match = "other grid")."""
    ops = ops.merge(ncu, on="signature", how="left")
    candidates = ncu.assign(kernel=ncu.signature.map(kernel_key))
    if "ncu_priority" in candidates:
        candidates = candidates[candidates.ncu_priority == candidates.groupby("kernel").ncu_priority.transform("min")]
    by_kernel = candidates.groupby("kernel")[["sol_pct", "sm_pct", "mem_pct"]].median()
    other = ops.sol_pct.isna() & (ops.kind == "kernel")
    ops["ncu_match"] = None
    ops.loc[ops.sol_pct.notna(), "ncu_match"] = "exact"
    for column in ("sol_pct", "sm_pct", "mem_pct"):
        ops.loc[other, column] = ops.signature[other].map(kernel_key).map(by_kernel[column])
    if "ncu_source" in ncu:
        sources = candidates.groupby("kernel").ncu_source.agg(lambda values: ", ".join(dict.fromkeys(values)))
        ops["ncu_source"] = ops["ncu_source"].astype(object)  # unmapped kernels leave NA, which an arrow string column rejects
        ops.loc[other, "ncu_source"] = ops.signature[other].map(kernel_key).map(sources)
    ops.loc[other & ops.sol_pct.notna(), "ncu_match"] = "other grid"
    return ops


def kernel_key(signature: str) -> str:
    """Mangled name and block size, without the grid."""
    name, _, block = signature.split("|")
    return f"{name}|{block}"


def number(text: str) -> float:
    return float(text.replace(",", "")) if text and text.strip() not in ("", "n/a") else float("nan")


# --- nsys timeline ----------------------------------------------------------------------------------------------


def timeline(db: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """GPU ops of the worker with their launch call, CPU phase path and step, plus the idle gap before each op."""
    db.execute(PHASE_SQL)
    tables = set(db.execute("SELECT table_name FROM information_schema.tables WHERE table_catalog = 't'").df().table_name)
    copies = "".join(COPY_SQL.format(kind=kind, copy_kind="copyKind" if kind == "memcpy" else "NULL::INT")
                     for kind in ["memcpy", "memset"] if COPY_TABLE.format(kind=kind.upper()) in tables)
    return db.execute(TIMELINE_SQL.replace("{copies}", copies)).df()


COPY_TABLE = "CUPTI_ACTIVITY_KIND_{kind}"  # nsys omits the table when there was no such op
COPY_SQL = """
    UNION ALL SELECT start, "end", deviceId, streamId, correlationId, globalPid, graphNodeId, '{kind}', '{kind}', '{kind}', bytes, {copy_kind} AS copy_kind
    FROM t.CUPTI_ACTIVITY_KIND_{kind}"""


PHASE_SQL = """
CREATE TEMP TABLE nvtx AS
SELECT n.start, n."end", n.eventType, n.globalTid, coalesce(n.text, s.value) AS text
FROM t.NVTX_EVENTS n LEFT JOIN t.StringIds s ON s.id = n.textId;

CREATE TEMP TABLE ranges AS
SELECT start, "end", globalTid, substr(text, 5) AS label FROM nvtx
WHERE text LIKE 'sol/%' AND text NOT LIKE 'sol/batch%' AND "end" IS NOT NULL;

-- A step is the CPU interval from one execute_model call to the next, so it includes sample_tokens and the draft.
-- Under TP every rank has its own worker thread, the analysis follows the first one.
CREATE TEMP TABLE steps AS
SELECT row_number() OVER w AS step, start, lead(start) OVER w AS next_start, globalTid
FROM ranges WHERE (label LIKE '%.execute_model' OR label = 'execute')  -- 'execute': prototype traces
    AND globalTid = (SELECT min(globalTid) FROM ranges WHERE label LIKE '%.execute_model' OR label = 'execute')
WINDOW w AS (ORDER BY start);

CREATE TEMP TABLE batches AS
SELECT s.step,
       try_cast(regexp_extract(n.text, 'reqs=(\\d+)', 1) AS INT) AS reqs,  -- NULL for "sol/batch unknown"
       try_cast(regexp_extract(n.text, 'tokens=(\\d+)', 1) AS INT) AS tokens,
       try_cast(regexp_extract(n.text, 'context=(\\d+)', 1) AS BIGINT) AS context,
       try_cast(regexp_extract(n.text, 'prefill=(\\d+)', 1) AS INT) AS prefill
FROM nvtx n JOIN steps s ON n.globalTid = s.globalTid AND n.start >= s.start AND n.start < s.next_start
WHERE n.text LIKE 'sol/batch%';
"""

TIMELINE_SQL = """
WITH worker AS (SELECT DISTINCT globalTid, globalTid // 16777216 * 16777216 AS globalPid FROM steps),
ops AS (
    SELECT k.start, k."end", k.deviceId, k.streamId, k.correlationId, k.globalPid, k.graphNodeId, 'kernel' AS kind,
           sn.value AS name, mn.value || '|(' || k.gridX || ', ' || k.gridY || ', ' || k.gridZ || ')|('
               || k.blockX || ', ' || k.blockY || ', ' || k.blockZ || ')' AS signature, NULL::BIGINT AS bytes, NULL::INT AS copy_kind
    FROM t.CUPTI_ACTIVITY_KIND_KERNEL k JOIN t.StringIds sn ON sn.id = k.shortName JOIN t.StringIds mn ON mn.id = k.mangledName
    {copies}
),
correlated AS (  -- all ops of the worker process, some graph nodes have no launch record (correlationId 0)
    SELECT o.*, r.start AS launch_start, r."end" AS launch_end, r.globalTid, api.value AS api
    FROM ops o
    JOIN (SELECT DISTINCT globalPid FROM worker) w USING (globalPid)
    LEFT JOIN t.CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = o.correlationId AND r.globalTid // 16777216 * 16777216 = o.globalPid
    LEFT JOIN t.StringIds api ON api.id = r.nameId
),
orphan_launches AS (  -- graph launches of the worker whose kernels carry no correlation id
    SELECT r.start AS o_start, r."end" AS o_end, r.globalTid AS o_tid, api.value AS o_api, w.globalPid
    FROM t.CUPTI_ACTIVITY_KIND_RUNTIME r JOIN worker w USING (globalTid) JOIN t.StringIds api ON api.id = r.nameId
    WHERE api.value LIKE 'cudaGraphLaunch%'
        AND r.correlationId NOT IN (SELECT correlationId FROM correlated WHERE launch_start IS NOT NULL)
),
launched AS (  -- an op without a launch record belongs to the latest orphan graph launch before it
    SELECT c.* REPLACE (coalesce(c.launch_start, o.o_start) AS launch_start, coalesce(c.launch_end, o.o_end) AS launch_end,
                        coalesce(c.globalTid, o.o_tid) AS globalTid, coalesce(c.api, o.o_api) AS api),
           c.launch_start IS NULL AND o.o_start IS NOT NULL AS inferred_launch
    FROM correlated c ASOF LEFT JOIN orphan_launches o ON c.globalPid = o.globalPid AND c.start >= o.o_start
),
launch_phases AS (  -- the nested sol ranges around each launch call, outermost first
    SELECT l.globalTid, l.launch_start, string_agg(g.label, '>' ORDER BY g.start) AS phase
    FROM (SELECT DISTINCT globalTid, launch_start FROM launched) l
    JOIN ranges g ON g.globalTid = l.globalTid AND g.start <= l.launch_start AND g."end" >= l.launch_start
    GROUP BY ALL
),
phased AS (
    SELECT l.*, s.step, p.phase
    FROM launched l
    LEFT JOIN launch_phases p USING (globalTid, launch_start)
    LEFT JOIN steps s ON s.globalTid = l.globalTid AND l.launch_start >= s.start
        AND (l.launch_start < s.next_start OR s.next_start IS NULL)
)
SELECT *,
    "end" - start AS dur_ns,
    max("end") OVER w AS prev_end,
    lag(correlationId) OVER (PARTITION BY deviceId ORDER BY start) AS prev_correlation
FROM phased
WINDOW w AS (PARTITION BY deviceId ORDER BY start, "end" ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
ORDER BY start
"""


def add_costs(ops: pd.DataFrame, hw: dict) -> pd.DataFrame:
    """Busy time counts each instant once across streams. The idle gap before an op is charged to that op."""
    ops = ops.copy()
    ops["busy_ns"] = (ops["end"] - ops[["start", "prev_end"]].max(axis=1)).clip(lower=0).fillna(ops.dur_ns)
    ops["gap_ns"] = (ops.start - ops.prev_end).clip(lower=0).fillna(0)
    ops["host_gap_ns"] = (ops.launch_end - ops.prev_end).clip(lower=0).clip(upper=ops.gap_ns).fillna(0)
    in_graph = ops.api.fillna("").str.startswith("cudaGraphLaunch") & (ops.correlationId == ops.prev_correlation)
    ops["gap_kind"] = "launch_latency"
    ops.loc[in_graph, "gap_kind"] = "in_graph"
    ops.loc[ops.host_gap_ns > 0, "gap_kind"] = "host"
    ops["category"] = [category(row.name, row.signature) for row in ops.itertuples()]
    no_launch = ops.launch_start.isna()
    ops["phase"] = ops.phase.fillna("(outside sol ranges)")
    ops.loc[no_launch, "phase"] = "(no launch record)"
    ops["step"] = ops.step.ffill()  # ops without a launch record belong to the step running before them
    ops.loc[no_launch, "gap_kind"] = "no_launch_record"
    return ops


def sol_sources(ops: pd.DataFrame, hw: dict) -> None:
    """Prefer semantic bounds, then profiler resource-demand references with explicit match quality.

    D2D copies read and write the payload. Host transfers need a host-link profile, not HBM bandwidth.
    Bounds stay unclipped so inconsistent assumptions remain visible instead of being forced to measured time.
    """
    copies = ops.kind != "kernel"
    defaults = {"model_quality": "unsupported", "model_reason": "no semantic mapping", "model_assumptions": "[]", "model_flops": "{}",
                "model_memory_ns": float("nan"), "model_compute_ns": float("nan"), "model_read_bytes": float("nan"), "model_write_bytes": float("nan")}
    for field, value in defaults.items():
        if field not in ops:
            ops[field] = value
    ops["sol_source"] = None
    modeled = ~copies & (ops.model_sol_ns > 0)
    ops["sol_ns"] = ops.model_sol_ns.where(modeled)
    ops.loc[modeled, "sol_source"] = "semantic: " + ops.model_family[modeled]
    if "copy_kind" not in ops:
        ops["copy_kind"] = float("nan")
    for (kind, direction, payload), group in ops[copies].groupby(["kind", "copy_kind", "bytes"], dropna=False):
        operation = "memset" if kind == "memset" else "device_to_device" if direction == 8 else "unknown_link"
        value = estimate("copy", {"kind": operation, "bytes": payload}, hardware_profile({}, hw))
        for field, entry in estimate_record(value).items():
            ops.loc[group.index, field] = entry
        ops.loc[group.index, "sol_ns"] = value.bound_ns
        if value.bound_ns is not None:
            ops.loc[group.index, "sol_source"] = "semantic: " + operation
    ops["sol_quality"] = ops.model_quality
    ops["sol_reason"] = ops.model_reason
    ops["sol_assumptions"] = ops.model_assumptions
    ops["sol_memory_ns"] = ops.model_memory_ns
    ops["sol_compute_ns"] = ops.model_compute_ns
    exact = ops.get("ncu_match", pd.Series(None, index=ops.index)).eq("exact")
    missing = pd.Series(float("nan"), index=ops.index)
    # NCU's --kernel-id filter keeps only the first launch of each (name, grid, block) signature across the whole
    # profiled run: a GEMM template reused at very different token counts (persistent-grid kernels do not size
    # their grid from the problem) broadcasts that one launch's bytes to every later occurrence of the signature.
    # A measurement that moves far more than our own no-cache ceiling allows is not a slow kernel, it is someone
    # else's launch, so the match is discarded rather than shown as a cross-check or used as a fallback SOL.
    model_bytes = ops.model_read_bytes.fillna(0) + ops.model_write_bytes.fillna(0)
    implausible = exact & (model_bytes > 0) & (ops.get("dram_bytes", missing) > model_bytes * 3)
    for column in ("sol_pct", "sm_pct", "mem_pct", "dram_bytes", "ncu_ns", "kernel_sol_ns", "dram_sol_ns"):
        if column in ops:
            ops.loc[implausible, column] = float("nan")
    ops.loc[implausible, "ncu_match"] = "exact signature, discarded: implausible vs semantic work"
    exact = exact & ~implausible
    duration = ops.get("ncu_ns", missing).where(exact, ops.dur_ns)
    # Vectorize the counter reference over large traces. The scalar policy is tested in benchmarking.roofline.
    compute = duration * ops.get("sm_pct", missing) / 100
    memory = duration * ops.get("mem_pct", missing) / 100
    reference = pd.concat([compute.where(compute > 0), memory.where(memory > 0)], axis=1).max(axis=1)
    fallback = (ops.kind == "kernel") & ops.sol_ns.isna() & (duration > 0) & (reference > 0)
    ops.loc[fallback, "sol_ns"] = reference[fallback]
    ops.loc[fallback, "sol_memory_ns"] = memory[fallback]
    ops.loc[fallback, "sol_compute_ns"] = compute[fallback]
    for is_exact, quality in ((True, "counter-reference"), (False, "inferred-counter-reference")):
        selected = fallback & (exact == is_exact)
        ops.loc[selected, "sol_source"] = "ncu reference: exact signature" if is_exact else "ncu reference: transferred utilization (other grid)"
        ops.loc[selected, "sol_quality"] = quality
        ops.loc[selected, "sol_reason"] = "resource-demand reference for measured work, excludes latency"
        ops.loc[selected, "sol_assumptions"] = json.dumps(["separate NCU replay, cache and launch conditions can differ"] + ([] if is_exact else ["same kernel and block, utilization transferred to this duration"]))


def category(name: str, signature: str = "") -> str:
    lowered = classification_name(name, signature).lower()
    return next((label for label, pattern in KERNEL_CATEGORIES if re.search(pattern, lowered)), "other")


# --- Steps ------------------------------------------------------------------------------------------------------


def step_table(db: duckdb.DuckDBPyConnection, ops: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Steady steps: pure decode with the most common request count, without the first and last step of the capture.
    Without batch marks every step but the first and last counts, and the rule says so."""
    costs = ops.groupby("step").agg(busy_ns=("busy_ns", "sum"), gap_ns=("gap_ns", "sum"), ops=("start", "size"))
    costs["cost_ns"] = costs.busy_ns + costs.gap_ns
    query = "SELECT s.step, s.start AS cpu_start, s.next_start AS cpu_end, b.* EXCLUDE (step) FROM steps s LEFT JOIN batches b USING (step)"
    steps = db.execute(query).df()
    if len(steps) < 3:
        raise SystemExit(f"{len(steps)} steps in the trace: the probe found no execute_model() or the capture was empty")
    steps = steps.merge(costs, left_on="step", right_index=True)
    inner = (steps.step > steps.step.min()) & (steps.step < steps.step.max())
    decode = steps.prefill == 0
    if not decode.any():
        steps["steady"] = inner
        return steps, "all steps (no batch shapes recorded)"
    mode_reqs = steps[decode].reqs.mode().iloc[0]
    steps["steady"] = inner & decode & (steps.reqs == mode_reqs)
    return steps, f"pure decode steps with {mode_reqs} requests"


def measurement_ops(db: duckdb.DuckDBPyConnection, ops: pd.DataFrame, level: str) -> pd.DataFrame:
    """Separate workloads from a shared trace before computing the modal request count and steady steps."""
    bounds = measurement_bounds(db, level)
    if bounds is None:
        return ops
    selected = db.execute("SELECT step FROM steps WHERE start>=? AND next_start<=?", list(bounds)).df().step
    return ops[ops.step.isin(selected)].copy()


def measured_steps(db: duckdb.DuckDBPyConnection, steps: pd.DataFrame, rule: str, level: str | None = None) -> tuple[pd.DataFrame, str]:
    """Shape traces include startup for graph provenance, while reported timings use the explicit workload window."""
    bounds = measurement_bounds(db, level)
    if bounds is None:
        return steps, rule
    start, end = bounds
    result = steps.copy()
    result["steady"] = result.steady & (result.cpu_start >= start) & (result.cpu_end <= end)
    if not result.steady.any():
        raise ValueError("no steady steps inside the NSYS shape measurement window")
    return result, rule + ", inside annotated measurement window"


def measurement_bounds(db: duckdb.DuckDBPyConnection, level: str | None = None) -> tuple[int, int] | None:
    suffix = f" c={int(level[1:])}" if level and re.fullmatch(r"c\d+", level) else ""
    query = """SELECT min(start) FILTER (WHERE text=?), max(start) FILTER (WHERE text=?) FROM nvtx
        WHERE globalTid // 16777216 IN (SELECT DISTINCT globalTid // 16777216 FROM steps)"""
    start, end = db.execute(query, ["sol/measurement_start" + suffix, "sol/measurement_end" + suffix]).fetchone()
    if start is None and end is None and suffix:
        start, end = db.execute(query, ["sol/measurement_start", "sol/measurement_end"]).fetchone()
    if start is None and end is None:
        tagged = db.execute("SELECT count(*) FROM nvtx WHERE text LIKE 'sol/measurement_start c=%'").fetchone()[0]
        if tagged:
            raise ValueError(f"no measurement window for {level}")
        return None
    if start is None or end is None or start >= end:
        raise ValueError("incomplete NSYS shape measurement window")
    return int(start), int(end)


def execution_shapes(db: duckdb.DuckDBPyConnection, steps: pd.DataFrame, config: Path) -> pd.DataFrame:
    """Execution marks come after input preparation. Old traces retain scheduled counts and declare the gap."""
    spec = yaml.safe_load(config.read_text()).get("serve_args", []) if config.exists() else []
    av = any('"enable_adaptive_verification":true' in arg.replace(" ", "").lower() for arg in spec)
    marks = db.execute("""SELECT s.step, substr(n.text, 15) AS payload FROM nvtx n JOIN steps s
        ON n.globalTid=s.globalTid AND n.start>=s.start AND n.start<s.next_start WHERE n.text LIKE 'sol/execution %'""").df()
    records = {row.step: json.loads(row.payload) for row in marks.itertuples()}
    result = steps.copy()
    for field in ("actual_tokens", "padded_tokens", "query_lengths", "context_lengths", "logits_tokens"):
        result[field] = [records.get(step, {}).get(field) for step in result.step]
    result["av_enabled"] = av
    result["query_shape_source"] = ["recorded execution" if step in records else "unrecorded AV execution" if av else "scheduler" for step in result.step]
    return result


# --- Analytic SOL -----------------------------------------------------------------------------------------------


def world_size(config: Path) -> int:
    """GPUs per server from the run's config, for the communication model."""
    return int(yaml.safe_load(config.read_text()).get("gpus", 1)) if config.exists() else 1


def model_info(path: Path) -> dict | None:
    """The sol_model_info() RPC result of rank 0, missing for runs from before the probe had it."""
    if not path.exists():
        return None
    info = json.loads(path.read_text())["results"][0]
    if "target" not in info:  # the first probe version had another layout
        if "target_weight_bytes" not in info:
            return None
        weights = lambda owner: {"dense_bytes": info[f"{owner}_weight_bytes"], "dense_elems": info[f"{owner}_weight_elems"],
                                 "expert_bytes": 0, "expert_elems": 0, "num_experts": 0}
        return {"target": weights("target"), "draft": weights("draft"), "top_k": 0,
                "target_kv_bytes_per_token": info["target_kv_bytes_per_token"],
                "draft_kv_bytes_per_token": info["draft_kv_bytes_per_token"],
                "target_state_bytes_per_req": 0, "draft_state_bytes_per_req": 0}
    return without_non_text(info)


NON_TEXT = re.compile(r"(^|\.)(vision|visual|audio)")


def without_non_text(info: dict) -> dict:
    """Weights a decode step must read, from the layer inventory when the probe reported one: the GEMM layers
    (linear, LM head) of the text model. That leaves out lookup tables that a forward reads only a few rows of
    (embeddings, n-gram memories), and vision or audio towers, which do not run in text decode. Norms and biases
    outside these layers are small."""
    if "layers" not in info:
        return info
    layers = {owner: [l for l in info["layers"][owner] if not NON_TEXT.search(l["name"])] for owner in info["layers"]}
    for owner, kept in layers.items():
        dense = [l for l in kept if l["kind"] in ("linear", "lm_head")]
        if info[owner] is not None:
            info[owner] = info[owner] | {"dense_bytes": sum(l["bytes"] for l in dense), "dense_elems": sum(l.get("logical_elems", l["elems"]) for l in dense)}
    return info | {"layers": layers}


def analytic_sol(steps: pd.DataFrame, ops: pd.DataFrame, info: dict | None, hw: dict) -> pd.DataFrame:
    """Partial work reference per forward, retaining separate roofs and declared assumptions.

    Target queries use recorded execution sizes when available. Draft queries are never borrowed from the target.
    Intermediate materialization and checkpoint writes are excluded, and MoE routing is an explicit assumption.
    """
    if info is None:
        return steps
    replays = ops[ops.phase.str.contains("propose") & ops.phase.str.contains("CUDAGraph.replay|>graph", regex=True)]
    forwards = replays.groupby("step").launch_start.nunique()
    draft_forwards = steps.step.map(forwards).fillna(1).clip(lower=1)
    target = [forward_estimate(info["target"], info, "target", row, hw) for row in steps.itertuples()]
    draft = [forward_estimate(info["draft"], info, "draft", row, hw) if info["draft"] else price("empty", Work(0), hardware_profile(info, hw)) for row in steps.itertuples()]
    result = steps.assign(target_sol_ns=[value.bound_ns for value in target],
                          draft_sol_ns=[value.bound_ns * count for value, count in zip(draft, draft_forwards)], draft_forwards=draft_forwards)
    result["analytic_sol_ns"] = result.target_sol_ns + result.draft_sol_ns
    result["analytic_memory_ns"] = [t.memory_ns + d.memory_ns * count for t, d, count in zip(target, draft, draft_forwards)]
    result["analytic_compute_ns"] = [t.compute_ns + d.compute_ns * count if t.compute_ns is not None and d.compute_ns is not None else None for t, d, count in zip(target, draft, draft_forwards)]
    result["analytic_quality"] = ["partial" if "partial" in (t.quality, d.quality) else "assumed" for t, d in zip(target, draft)]
    result["analytic_assumptions"] = [json.dumps(sorted(set(t.work.assumptions + d.work.assumptions))) for t, d in zip(target, draft)]
    return result


def forward_sol(weights: dict, info: dict, owner: str, steps: pd.DataFrame, hw: dict) -> pd.Series:
    return pd.Series([forward_estimate(weights, info, owner, row, hw).bound_ns for row in steps.itertuples()], index=steps.index)


def forward_estimate(weights: dict, info: dict, owner: str, step, hw: dict) -> Estimate:
    """Optimistic per-forward work reference. State persistence and unknown draft shapes are explicit."""
    moe = next((layer for layer in info.get("layers", {}).get(owner, []) if layer["kind"] == "moe"), {})
    k = moe.get("top_k") or info["top_k"] or 0
    experts = moe.get("global_experts") or weights["num_experts"]  # with expert parallelism the rank holds a share
    requests = float(step.reqs) if pd.notna(step.reqs) else 0
    touched = 1 - (1 - k / experts) ** requests if experts and k else 1.0  # fraction of experts read
    active = k / experts if experts and k else 1.0  # fraction of expert FLOPs per token
    tokens = query_tokens(step, owner)
    context = float(step.context) if pd.notna(step.context) else 0
    kv = info[f"{owner}_kv_bytes_per_token"] * (context + (tokens or 0))
    state = info.get(f"{owner}_state_bytes_per_req", 0) * requests
    reads = weights["dense_bytes"] + weights["expert_bytes"] * touched + kv + state
    flops = {info.get("activation_dtype", "bf16"): 2 * (weights["dense_elems"] + weights["expert_elems"] * active) * tokens} if tokens is not None else {}
    assumptions = ["weights and existing KV/state read once from device memory", "intermediate activations and checkpoint writes excluded",
                   "attention, recurrence and nonlinear arithmetic excluded"]
    assumptions.append("query shape unavailable, query-dependent work excluded" if tokens is None else "query-dependent work uses " + str(getattr(step, "query_shape_source", "scheduler")))
    if experts:
        assumptions.append("uniform expert routing, drafts of one request share experts")
    return price("forward", Work(reads, flops=flops, assumptions=tuple(assumptions), complete=False), hardware_profile(info, hw))


# --- Report -----------------------------------------------------------------------------------------------------


def has_kernel_sol(ops: pd.DataFrame) -> bool:
    """Whether kernels have a SOL estimate, from the analytic models or ncu. Copies always get one from their bytes."""
    return bool(((ops.kind == "kernel") & ops.sol_ns.notna()).any())



def waterfall_rows(ops: pd.DataFrame, steps: pd.DataFrame) -> list[tuple[str, float]]:
    """Components of the mean step cost in µs. They sum to the mean step cost."""
    n = len(steps)
    per_step = lambda value: value / n / 1e3  # noqa: E731, ns totals -> µs per step
    known = ops.sol_ns.notna()
    sol = per_step(ops.sol_ns[known].sum())
    rows = [
        ("Per-kernel SOL", sol),
        ("Kernel inefficiency (busy time above SOL)", per_step(ops.busy_ns[known].sum()) - sol),
        ("Busy time without a SOL estimate", per_step(ops.busy_ns[~known].sum())),
    ]
    if "analytic_sol_ns" in steps:
        # Per-kernel SOL splits the time above the analytic floor only when it covers (nearly) every kernel,
        # otherwise the two sums are not comparable and the busy time stays one block.
        analytic = steps.analytic_sol_ns.mean() / 1e3
        complete = ops.busy_ns[known].sum() >= 0.99 * ops.busy_ns.sum() and analytic <= sol <= per_step(ops.busy_ns.sum())
        split = [("Per-kernel SOL above analytic (fusion, extra traffic, small kernels)", sol - analytic)] + rows[1:2]
        whole = [("Busy time above analytic SOL", per_step(ops.busy_ns.sum()) - analytic)]
        rows = [("Analytic SOL (optimistic work reference)", analytic)] + (split if complete else whole)
    return rows + [(f"Idle: {kind}", per_step(group.gap_ns.sum())) for kind, group in ops.groupby("gap_kind")]


def sol_floor(ops: pd.DataFrame, steps: pd.DataFrame) -> tuple[str, float] | None:
    """The headline SOL in µs per step: the analytic one when the model info exists, else the per-kernel one."""
    if "analytic_sol_ns" in steps:
        return "analytic SOL", steps.analytic_sol_ns.mean() / 1e3
    return ("per-kernel SOL", waterfall_rows(ops, steps)[0][1]) if has_kernel_sol(ops) else None


def waterfall(ops: pd.DataFrame, steps: pd.DataFrame, hw: dict) -> str:
    n = len(steps)
    rows = waterfall_rows(ops, steps)
    total = steps.cost_ns.mean() / 1e3
    floor = sol_floor(ops, steps)
    drafts = steps.tokens.sum() / steps.reqs.sum() - 1
    lines = [
        f"# SOL waterfall, {n} steady steps, {steps.reqs.mean():.0f} reqs x {steps.tokens.sum() / steps.reqs.sum():.0f} "
        f"scheduled tokens on average ({drafts:.0f} drafts + 1 verify)",
        "",
        f"- GPU: {hw['gpu']}, peak DRAM BW {hw['bw'] / 1e12:.2f} TB/s",
        f"- Mean step cost {total:.1f} µs (median {steps.cost_ns.median() / 1e3:.1f} µs)"
        + (f", {total / floor[1]:.2f}x {floor[0]} ({floor[1]:.1f} µs)" if floor else ", no SOL estimate")
        + (f", median step {steps.cost_ns.median() / 1e3 / floor[1]:.2f}x" if floor else ""),
        *stall_lines(steps, floor, host_gaps(ops)),
        "",
        "| Component | µs/step | share |",
        "|---|---:|---:|",
        *[f"| {label} | {value:.1f} | {value / total:.1%} |" for label, value in rows],
        f"| **Total step cost** | **{total:.1f}** | |",
        "",
        "## By CPU phase of the launch",
        "",
        breakdown(ops, "phase", n),
        "",
        "## By kernel category",
        "",
        breakdown(ops, "category", n),
        "",
        "## Top kernels by lost time (cost minus SOL)",
        "",
        breakdown(ops, ["phase", "name"], n, limit=20),
        "",
        "## Host-bound idle by CPU phase",
        "",
        host_breakdown(ops, n),
    ]
    return "\n".join(lines) + "\n"


STALL_MIN_NS = 1e6  # a launch gap is microseconds, so 1 ms of idle means the queue ran dry


def host_gaps(ops: pd.DataFrame) -> pd.Series:
    """GPU idle the host caused, per step, without the in-graph and launch gaps every step has."""
    return ops[ops.gap_kind == "host"].groupby("step").gap_ns.sum()


def stall_summary(steps: pd.DataFrame, host: pd.Series | None = None) -> dict | None:
    """Steps where the host left the GPU idle long enough that the pipeline, not the kernel work, set the step cost."""
    idle = steps.gap_ns if host is None else host.reindex(steps.step).fillna(0).to_numpy()
    threshold = max(STALL_MIN_NS, 5 * float(pd.Series(idle).median()))
    stalled = idle > threshold
    if not stalled.any():
        return None
    rest = steps.cost_ns[~stalled]
    return {
        "count": int(stalled.sum()),
        "steps": len(steps),
        "threshold_us": threshold / 1e3,
        "time_share": float(steps.cost_ns[stalled].sum() / steps.cost_ns.sum()),
        "mean_cost_ns": float(rest.mean()) if len(rest) else float("nan"),
    }


def stall_check(steps: pd.DataFrame, host: pd.Series | None = None) -> str:
    stall = stall_summary(steps, host)
    if not stall:
        return "none"
    over_half = ", over half the steady steps" if stall["count"] * 2 > stall["steps"] else ""
    return (f"{stall['count']} of {stall['steps']} ({stall['time_share']:.1%} of the time, "
            f"host gap above {stall['threshold_us'] / 1e3:g} ms{over_half})")


def stall_lines(steps: pd.DataFrame, floor: tuple[str, float] | None, host: pd.Series | None = None) -> list[str]:
    """Markdown bullets that keep the all-steps mean honest: how much of it stalls, and the mean without."""
    stall = stall_summary(steps, host)
    if not stall:
        return []
    text = (f"- Stalls: {stall['count']} of {stall['steps']} steady steps leave the GPU idle above {stall['threshold_us'] / 1e3:g} ms "
            f"on the host and hold {stall['time_share']:.1%} of the measured time, where the pipeline, not the kernels, sets the step cost")  # fmt: skip
    if floor and not pd.isna(stall["mean_cost_ns"]):
        text += f". Without them the mean step is {stall['mean_cost_ns'] / 1e3:.1f} µs ({stall['mean_cost_ns'] / 1e3 / floor[1]:.2f}x {floor[0]})"
    return [text]


def breakdown_table(ops: pd.DataFrame, by, n: int, limit: int | None = None) -> pd.DataFrame:
    """Unknown operations never masquerade as running at SOL. Incomplete groups retain measured times only."""
    frame = ops.assign(cost=ops.busy_ns + ops.gap_ns, sol=ops.sol_ns, known=ops.sol_ns.notna())
    table = frame.groupby(by).agg(ops=("start", "size"), cost=("cost", "sum"), busy=("busy_ns", "sum"),
                                  sol=("sol", "sum"), idle=("gap_ns", "sum"), known=("known", "sum"))  # fmt: skip
    mapped = frame[frame.known].groupby(by).busy_ns.sum().reindex(table.index).fillna(0) / table.busy
    table.loc[table.known != table.ops, "sol"] = float("nan")
    table = table.drop(columns="known")
    table = table / [n, 1e3 * n, 1e3 * n, 1e3 * n, 1e3 * n]
    table["lost"] = table.cost - table.sol
    table["x_sol"] = table.cost / table.sol
    table["mapped"] = mapped
    return table.assign(rank_cost=table.cost - table.sol.fillna(0)).sort_values("rank_cost", ascending=False).head(limit)


def breakdown(ops: pd.DataFrame, by, n: int, limit: int | None = None) -> str:
    table = breakdown_table(ops, by, n, limit)
    header = "| " + " | ".join(table.index.names) + " | ops/step | cost us | busy us | SOL us | idle us | lost us | x SOL | SOL coverage |"
    lines = [header, "|" + "---|" * len(table.index.names) + "---:|" * 8]
    for key, row in table.iterrows():
        keys = key if isinstance(key, tuple) else (key,)
        x_sol = row.x_sol if pd.isna(row.x_sol) else min(row.x_sol, 99)  # >99x only for tiny copies, NA stays NA
        values = " | ".join(f"{value:.1f}" if pd.notna(value) else "n/a" for value in (row.ops, row.cost, row.busy, row.sol, row.idle, row.lost, x_sol))
        lines.append("| " + " | ".join(str(k)[:80] for k in keys) + f" | {values} | {int(row.mapped * 100)}% |")
    return "\n".join(lines)


def host_breakdown(ops: pd.DataFrame, n: int) -> str:
    host = ops[ops.host_gap_ns > 0].groupby("phase").host_gap_ns.agg(["size", "sum"]).sort_values("sum", ascending=False)
    lines = ["| phase | gaps/step | host-bound us/step |", "|---|---:|---:|"]
    lines += [f"| {phase} | {row['size'] / n:.1f} | {row['sum'] / n / 1e3:.1f} |" for phase, row in host.iterrows()]
    lines += ["", "Host-bound idle is the attributable part of every gap kind, so it is not the waterfall's `Idle: host` bucket."]
    return "\n".join(lines)


def representative_steps(steps: pd.DataFrame, count: int) -> list[int]:
    """Consecutive steady steps whose mean cost is closest to the median step cost."""
    steady = steps[steps.steady].set_index("step").cost_ns
    windows = steady.rolling(count).mean().dropna()
    end = (windows - steady.median()).abs().idxmin()
    return list(range(end - count + 1, end + 1))


def cpu_ranges(db: duckdb.DuckDBPyConnection, selected: pd.DataFrame) -> pd.DataFrame:
    """sol/* NVTX ranges of the worker thread that overlap the selected ops or their launches."""
    t0 = min(selected.launch_start.min(), selected.start.min())
    t1 = max(selected.launch_end.max(), selected["end"].max())
    query = 'SELECT * FROM ranges WHERE globalTid IN (SELECT globalTid FROM steps) AND start < ? AND "end" > ? ORDER BY start'
    return db.execute(query, [int(t1), int(t0)]).df()


def perfetto_trace(db: duckdb.DuckDBPyConnection, ops: pd.DataFrame, step_ids: list[int]) -> dict:
    """Chrome trace JSON for ui.perfetto.dev: CPU phase ranges, GPU ops colored by x SOL, and idle gaps."""
    selected = ops[ops.step.isin(step_ids)]
    t0 = min(selected.launch_start.min(), selected.start.min())
    t1 = max(selected.launch_end.max(), selected["end"].max())
    events = [meta(1, "CPU worker"), meta(2, "GPU")]
    window = [int(t1), int(t0)]
    ranges = cpu_ranges(db, selected)
    events += [slice_event(1, 1, r.label, r.start - t0, r.end - r.start, {}) for r in ranges.itertuples()]
    events += [slice_event(1, 2, r.api, r.start - t0, r.end - r.start, {}) for r in runtime_calls(db, window)]
    for op in selected.itertuples():
        x_sol = op.dur_ns / op.sol_ns if op.sol_ns > 0 else None  # NaN compares False
        args = {"phase": op.phase, "step": op.step, "category": op.category, "busy_us": op.busy_ns / 1e3,
                "sol_us": none_if_nan(op.sol_ns, 1e3), "x_sol": x_sol, "sol_pct": none_if_nan(op.sol_pct, 1)}  # fmt: skip
        color = next(c for limit, c in SOL_COLORS if x_sol < limit) if x_sol else "grey"
        events.append(slice_event(2, op.streamId, op.name, op.start - t0, op.dur_ns, args, color))
        if op.gap_ns > 0:
            gap = {"kind": op.gap_kind, "host_us": op.host_gap_ns / 1e3, "phase": op.phase}
            events.append(slice_event(2, 0, f"idle:{op.gap_kind}", op.prev_end - t0, op.gap_ns, gap, "bad"))
    return {"traceEvents": events, "displayTimeUnit": "ns"}


def runtime_calls(db: duckdb.DuckDBPyConnection, window: list[int], min_ns: int = 5000):
    """CUDA API calls of the worker thread longer than min_ns, e.g. graph launches and synchronizations."""
    query = """SELECT r.start, r."end", s.value AS api FROM t.CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN t.StringIds s ON s.id = r.nameId WHERE r.globalTid IN (SELECT globalTid FROM steps)
        AND r.start < ? AND r."end" > ? AND r."end" - r.start >= ?"""
    return db.execute(query, window + [min_ns]).df().itertuples()


def slice_event(pid, tid, name, start_ns, dur_ns, args, color=None) -> dict:
    event = {"ph": "X", "pid": pid, "tid": int(tid), "name": name, "ts": start_ns / 1e3, "dur": dur_ns / 1e3}
    event["args"] = args
    if color:
        event["cname"] = color
    return event


def meta(pid: int, name: str) -> dict:
    return {"ph": "M", "pid": pid, "name": "process_name", "args": {"name": name}}


def none_if_nan(value, scale: float):
    return None if pd.isna(value) else value / scale


if __name__ == "__main__":
    main()
