"""SOL waterfall of steady-state spec-decode steps, from the nsys sqlite export and the ncu raw CSV of one run.

Every GPU op of a step gets a time cost: its busy time plus the GPU idle gap before it. The waterfall splits the
mean step cost into per-kernel SOL, kernel inefficiency, busy time without a SOL estimate, and idle gaps by cause.
A kernel's SOL comes from the first source in SOL_SOURCES that knows it. Today that is ncu's busiest-unit
throughput, which needs no model of the operation. Op models (FLOPs and bytes from shapes) can be added as sources.
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

from kernel_models import classification_name, kernel_models

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
    parser.add_argument("--ncu", type=Path, help="ncu CSV to use, e.g. from another run of the same workload")
    parser.add_argument("--trace-steps", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    hw = hardware(args.run_dir / "nsys/trace.sqlite") | {"world_size": world_size(args.run_dir.parent / "config.yaml")}
    db = duckdb.connect()
    db.execute(f"ATTACH '{args.run_dir / 'nsys/trace.sqlite'}' AS t (TYPE sqlite, READ_ONLY)")
    ops = timeline(db)
    ops = match_ncu(ops, ncu_sol(args.ncu or args.run_dir / "ncu/kernels.csv", hw))
    ops = add_costs(ops, hw)
    steps, steady_rule = step_table(db, ops)
    info = model_info(args.run_dir / "nsys/model-info.json")
    steps = analytic_sol(steps, ops, info, hw)
    with_layers = info and "layers" in info  # runs from before the probe reported its layer inventory have none
    ops, mapping = kernel_models(ops, steps, info, hw) if with_layers else (ops.assign(model_sol_ns=float("nan"), model_family=None), {})
    sol_sources(ops, hw)
    steady = steps[steps.steady]
    steady_ops = ops[ops.step.isin(steady.step)]
    ops.to_parquet(args.output / "ops.parquet")
    steps.to_parquet(args.output / "steps.parquet")
    summary = waterfall(steady_ops, steady, hw)
    (args.output / "summary.md").write_text(summary)
    shown = representative_steps(steps, args.trace_steps)
    cpu_ranges(db, ops[ops.step.isin(shown)]).to_parquet(args.output / "ranges.parquet")
    meta = {"hw": hw, "shown_steps": shown, "checks": checks(db, ops, steady_ops, steps, steady_rule) | mapping}
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
    }


# --- Inputs -----------------------------------------------------------------------------------------------------


def hardware(trace: Path) -> dict:
    """ncu percentages are relative to the same peaks, nsys records the DRAM peak for copies."""
    with sqlite3.connect(trace) as db:
        name, peak_bw = db.execute("SELECT name, max(memoryBandwidth) FROM TARGET_INFO_GPU").fetchone()
    return {"bw": peak_bw, "gpu": name}


def ncu_sol(path: Path, hw: dict) -> pd.DataFrame:
    """Per kernel signature: time at the SOL of its busiest unit, measured DRAM bytes, and the SOL percentage."""
    if not path.exists():
        print(f"no ncu CSV at {path}, kernels get no SOL")
        columns = ["ncu_ns", "sol_pct", "kernel_sol_ns", "dram_bytes", "dram_sol_ns"]
        return pd.DataFrame({"signature": pd.Series(dtype=str), **{c: pd.Series(dtype=float) for c in columns}})
    rows = list(csv.DictReader(path.open()))
    rows = [r for r in rows[1:] if r.get("ID", "").strip().isdigit()]  # row 0 holds the units
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
    columns = ["ncu_ns", "sol_pct", "kernel_sol_ns", "dram_bytes", "dram_sol_ns"]
    return frame.groupby("signature", as_index=False)[columns].median().assign(ncu_samples=frame.groupby("signature").size().values)


def match_ncu(ops: pd.DataFrame, ncu: pd.DataFrame) -> pd.DataFrame:
    """Exact match on the signature first. Kernels whose grid depends on the data (MoE GEMMs over routed tokens,
    per-step token counts with AV) rarely repeat a grid, those take the median SOL % of the same kernel and block
    size, applied to their own duration (ncu_match = "other grid")."""
    ops = ops.merge(ncu, on="signature", how="left")
    by_kernel = ncu.assign(kernel=ncu.signature.map(kernel_key)).groupby("kernel").sol_pct.median()
    other = ops.sol_pct.isna() & (ops.kind == "kernel")
    ops["ncu_match"] = None
    ops.loc[ops.sol_pct.notna(), "ncu_match"] = "exact"
    ops.loc[other, "sol_pct"] = ops.signature[other].map(kernel_key).map(by_kernel)
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
    copies = "".join(COPY_SQL.format(kind=kind) for kind in ["memcpy", "memset"] if COPY_TABLE.format(kind=kind.upper()) in tables)
    return db.execute(TIMELINE_SQL.replace("{copies}", copies)).df()


COPY_TABLE = "CUPTI_ACTIVITY_KIND_{kind}"  # nsys omits the table when there was no such op
COPY_SQL = """
    UNION ALL SELECT start, "end", deviceId, streamId, correlationId, globalPid, graphNodeId, '{kind}', '{kind}', '{kind}', bytes
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
               || k.blockX || ', ' || k.blockY || ', ' || k.blockZ || ')' AS signature, NULL::BIGINT AS bytes
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
    """Fills sol_ns and sol_source from the first estimate that knows the op. An op without any keeps sol_ns NaN
    and shows up as "without a SOL estimate".

    1. copies: bytes / peak DRAM bandwidth.
    2. kernels: the analytic model of kernel_models.py, from the layer it belongs to and the step's shape.
    3. kernels: ncu duration x max(SM %, memory %), the time at the peak of the kernel's busiest unit (measured
       under ncu replay). It needs no model, but cannot tell redundant work from useful work.
    """
    copies = ops.kind != "kernel"
    ops.loc[copies, "kernel_sol_ns"] = ops.bytes[copies] / hw["bw"] * 1e9
    other_grid = ops.ncu_match == "other grid"
    ops.loc[other_grid, "kernel_sol_ns"] = ops.dur_ns[other_grid] * ops.sol_pct[other_grid] / 100
    modeled = ~copies & ops.model_sol_ns.notna()
    ops["sol_source"] = None
    ops.loc[copies & ops.kernel_sol_ns.notna(), "sol_source"] = "bytes / peak DRAM BW"
    ops.loc[~copies & ops.kernel_sol_ns.notna(), "sol_source"] = "ncu busiest unit"
    ops.loc[other_grid & ops.kernel_sol_ns.notna(), "sol_source"] = "ncu busiest unit, same kernel with another grid"
    ops.loc[modeled, "sol_source"] = "analytic: " + ops.model_family[modeled]
    ops["sol_ns"] = ops.model_sol_ns.where(modeled, ops.kernel_sol_ns).clip(upper=ops.busy_ns)


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
        return None
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
            info[owner] = info[owner] | {"dense_bytes": sum(l["bytes"] for l in dense), "dense_elems": sum(l["elems"] for l in dense)}
    return info | {"layers": layers}


def analytic_sol(steps: pd.DataFrame, ops: pd.DataFrame, info: dict | None, hw: dict) -> pd.DataFrame:
    """A lower bound per step that needs no profiler: every forward moves at least its weights, its KV and recurrent
    state, and does 2 x active params x tokens FLOPs, at peak DRAM bandwidth or the measured BF16 GEMM rate.

    - target: one forward over the step's tokens (verify).
    - draft: one forward per CUDA graph replay launched inside propose() (at least one), each priced like a forward
      over all the step's tokens, which overstates draft FLOPs, but decode is bound by bytes.
    - MoE: a step reads only the experts its requests route to, E * (1 - (1 - k/E)^R) for R requests under uniform
      routing. The verified and draft tokens of one request route alike, counting them separately overshoots.
    Activations and attention FLOPs are left out, so the bound is optimistic.
    """
    if info is None:
        return steps
    replays = ops[ops.phase.str.contains("propose") & ops.phase.str.contains("CUDAGraph.replay|>graph", regex=True)]
    forwards = replays.groupby("step").launch_start.nunique()
    draft_forwards = steps.step.map(forwards).fillna(1).clip(lower=1)
    target = forward_sol(info["target"], info, "target", steps, hw)
    draft = forward_sol(info["draft"], info, "draft", steps, hw) * draft_forwards if info["draft"] else 0
    return steps.assign(target_sol_ns=target, draft_sol_ns=draft, analytic_sol_ns=target + draft, draft_forwards=draft_forwards)


def forward_sol(weights: dict, info: dict, owner: str, steps: pd.DataFrame, hw: dict) -> pd.Series:
    """ns per forward for every step, the larger of the byte time and the FLOP time."""
    moe = next((layer for layer in info.get("layers", {}).get(owner, []) if layer["kind"] == "moe"), {})
    k = moe.get("top_k") or info["top_k"] or 0
    experts = moe.get("global_experts") or weights["num_experts"]  # with expert parallelism the rank holds a share
    touched = 1 - (1 - k / experts) ** steps.reqs if experts and k else 1.0  # fraction of experts read
    active = k / experts if experts and k else 1.0  # fraction of expert FLOPs per token
    state = info[f"{owner}_kv_bytes_per_token"] * (steps.context + steps.tokens) + 2 * info[f"{owner}_state_bytes_per_req"] * steps.reqs
    memory = (weights["dense_bytes"] + weights["expert_bytes"] * touched + state) / hw["bw"]
    compute = 2 * (weights["dense_elems"] + weights["expert_elems"] * active) * steps.tokens / info["gemm_flops_per_s"]
    return pd.concat([memory, compute], axis=1).max(axis=1) * 1e9


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
        complete = ops.busy_ns[known].sum() >= 0.99 * ops.busy_ns.sum()
        split = [("Per-kernel SOL above analytic (fusion, extra traffic, small kernels)", sol - analytic)] + rows[1:2]
        whole = [("Busy time above analytic SOL", per_step(ops.busy_ns.sum()) - analytic)]
        rows = [("Analytic SOL (weights, KV and state once per forward)", analytic)] + (split if complete else whole)
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
    lines = [
        f"# SOL waterfall, {n} steady steps, {steps.reqs.mean():.0f} reqs x {steps.tokens.mean():.0f} tokens on average",
        "",
        f"- GPU: {hw['gpu']}, peak DRAM BW {hw['bw'] / 1e12:.2f} TB/s",
        f"- Mean step cost {total:.1f} µs (median {steps.cost_ns.median() / 1e3:.1f} µs)"
        + (f", {total / floor[1]:.2f}x {floor[0]} ({floor[1]:.1f} µs)" if floor else ", no SOL estimate"),
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


def breakdown_table(ops: pd.DataFrame, by, n: int, limit: int | None = None) -> pd.DataFrame:
    """Per-step us by group, sorted by lost time. Ops without ncu data count as running at SOL."""
    frame = ops.assign(cost=ops.busy_ns + ops.gap_ns, sol=ops.sol_ns.fillna(ops.busy_ns))
    table = frame.groupby(by).agg(ops=("start", "size"), cost=("cost", "sum"), busy=("busy_ns", "sum"),
                                  sol=("sol", "sum"), idle=("gap_ns", "sum"))  # fmt: skip
    table = table / [n, 1e3 * n, 1e3 * n, 1e3 * n, 1e3 * n]
    table["lost"] = table.cost - table.sol
    table["x_sol"] = table.cost / table.sol
    return table.sort_values("lost", ascending=False).head(limit)


def breakdown(ops: pd.DataFrame, by, n: int, limit: int | None = None) -> str:
    table = breakdown_table(ops, by, n, limit)
    header = "| " + " | ".join(table.index.names) + " | ops/step | cost us | busy us | SOL us | idle us | lost us | x SOL |"
    lines = [header, "|" + "---|" * len(table.index.names) + "---:|" * 7]
    for key, row in table.iterrows():
        keys = key if isinstance(key, tuple) else (key,)
        values = f"{row.ops:.1f} | {row.cost:.1f} | {row.busy:.1f} | {row.sol:.1f} | {row.idle:.1f} | {row.lost:.1f} | {min(row.x_sol, 99):.1f}"  # >99x only for tiny copies
        lines.append("| " + " | ".join(str(k)[:80] for k in keys) + f" | {values} |")
    return "\n".join(lines)


def host_breakdown(ops: pd.DataFrame, n: int) -> str:
    host = ops[ops.host_gap_ns > 0].groupby("phase").host_gap_ns.agg(["size", "sum"]).sort_values("sum", ascending=False)
    lines = ["| phase | gaps/step | us/step |", "|---|---:|---:|"]
    lines += [f"| {phase} | {row['size'] / n:.1f} | {row['sum'] / n / 1e3:.1f} |" for phase, row in host.iterrows()]
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
        x_sol = op.busy_ns / op.sol_ns if op.sol_ns > 0 else None  # NaN compares False
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
