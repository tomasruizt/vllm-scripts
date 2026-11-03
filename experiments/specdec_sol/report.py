"""Offline HTML report of one SOL run: one tab per concurrency level, built from sol.py outputs and run artifacts.

    report.py results/<RUN_ID> report/<RUN_ID>   # writes report/<RUN_ID>/index.html
"""

import argparse
import html
import json
import math
import shlex
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from benchmarking.aiperf import server_total  # noqa: E402
from reporting.model_tabs import model_tabs  # noqa: E402
from sol import breakdown_table, has_kernel_sol, host_gaps, sol_floor, stall_summary, waterfall_rows  # noqa: E402

THEME = (REPO / "reporting/b200-theme.css").read_text()
NO_KERNEL_SOL_NOTE = (
    '<p class="metric-note">No per-kernel semantic or counter reference for this run. '
    "The timeline is uncolored, and the tables show only measured time.</p>"
)
COMPONENT_COLORS = {  # by label prefix, so a component keeps its color when others are missing
    "Analytic": "#235cca", "Per-kernel SOL above": "#7aa2e8", "Per-kernel SOL": "#235cca", "Kernel inefficiency": "#e4ad31",
    "Busy time above": "#e4ad31", "Busy time without": "#9ca3af",
    "Idle: host": "#b91c1c", "Idle: in_graph": "#ef6b6b", "Idle: launch": "#f5a3a3",
}  # fmt: skip
LANE = 27
GAP = 6  # between lanes
STYLE = """
.timeline { display: flex; background: white; border: 1px solid #dce3ed; border-radius: 12px; overflow: hidden; }
.timeline svg { display: block; font-family: system-ui, sans-serif; }
.timeline .lanes { flex: none; border-right: 1px solid #e7ecf3; }
.timeline .scroller { flex: 1; overflow-x: auto; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; margin-top: 12px; }
.legend span::before { content: ""; display: inline-block; width: 12px; height: 12px; margin-right: 6px;
  vertical-align: -1px; border-radius: 2px; background: var(--c); }
td.name, th.name { text-align: left; font-family: ui-monospace, monospace; font-size: 12px; }
td.name { white-space: normal; overflow-wrap: anywhere; min-width: 220px; }
td.name.phase { white-space: nowrap; width: 1%; }  /* shrink to content, break only after each " > " */
.stacked-bar { display: flex; height: 36px; margin-top: 44px; }
.segment { position: relative; min-width: 2px; }
.segment:hover { outline: 2px solid #111827; z-index: 1; }
.segment:hover::after { content: attr(data-tip); position: absolute; bottom: 44px; white-space: pre; font-size: 13px;
  background: #111827; color: white; padding: 6px 10px; border-radius: 6px; pointer-events: none; }
.segment.left:hover::after { left: 0; }
.segment.right:hover::after { right: 0; }
.hit:hover rect:first-child { stroke: #111827; stroke-width: 1.5; }
.hit.selected > rect:first-child { stroke: #235cca; stroke-width: 1.2; stroke-opacity: .65; }
.detail-tip { position: fixed; z-index: 10; pointer-events: none; white-space: pre; font-size: 12px; line-height: 1.5;
  background: #111827; color: white; padding: 8px 10px; border-radius: 6px; max-width: 640px; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; }
.plot { cursor: crosshair; user-select: none; }
.measure { fill: rgba(35, 92, 202, .12); stroke: #235cca; stroke-width: 1; pointer-events: none; }
.measure-label { font-size: 13px; font-weight: 600; fill: #1e3a8a; text-anchor: middle; pointer-events: none;
  paint-order: stroke; stroke: white; stroke-width: 4px; }
"""
# Every tab embeds this script, so it installs itself once: one tooltip for the page, and zoom on every timeline.
TIMELINE_SCRIPT = r"""<script>
if (!window.detailTip) {
  const MAX_PX_PER_US = 200;
  const layout = (view, scale) => {
    view.scale = scale;
    const svg = view.querySelector(".plot");
    svg.setAttribute("width", view.span * scale + 10);
    for (const el of view.boxes) {
      el.setAttribute("x", +el.dataset.a * scale);
      el.setAttribute("width", Math.max((+el.dataset.b - +el.dataset.a) * scale, 0.6));
    }
    placeLabels(view);
    drawAxis(view, svg, scale);
    drawMeasure(view);
  };
  // Drag on the plot to measure a time span, a click without dragging clears it.
  const drawMeasure = (view) => {
    const m = view.measure;
    const a = m && Math.min(m.a, m.b);
    const b = m && Math.max(m.a, m.b);
    view.selectedKernels = view.kernels.filter((el) => {
      const selected = !!m && b > a && +el.dataset.b > a && +el.dataset.a < b;
      el.classList.toggle("selected", selected);
      return selected;
    });
    view.measureBox.style.display = view.measureText.style.display = m ? "" : "none";
    if (!m) return;
    const x = a * view.scale;
    const w = (b - a) * view.scale;
    view.measureBox.setAttribute("x", x);
    view.measureBox.setAttribute("width", w);
    view.measureText.setAttribute("x", x + w / 2);
    view.measureText.textContent = `${(b - a).toFixed(1)} µs · ${view.selectedKernels.length} kernels · cmd+c to copy`;
  };
  const timeAt = (view, event) => Math.max(0, Math.min(view.span, (event.clientX - view.plot.getBoundingClientRect().left) / view.scale));
  const setupMeasure = (view) => {
    const ns = "http://www.w3.org/2000/svg";
    view.plot = view.querySelector(".plot");
    view.measureBox = document.createElementNS(ns, "rect");
    view.measureBox.setAttribute("class", "measure");
    view.measureBox.setAttribute("y", 14);
    view.measureBox.setAttribute("height", view.plot.getAttribute("height") - 22);
    view.measureText = document.createElementNS(ns, "text");
    view.measureText.setAttribute("class", "measure-label");
    view.measureText.setAttribute("y", 30);
    view.plot.append(view.measureBox, view.measureText);
    view.plot.addEventListener("mousedown", (event) => {
      if (event.button !== 0 || !view.scale) return;
      event.preventDefault();  // no text selection while dragging
      window.getSelection()?.removeAllRanges();
      if (window.measuredView && window.measuredView !== view) {
        window.measuredView.measure = null;
        drawMeasure(window.measuredView);
      }
      view.measure = { a: timeAt(view, event), b: timeAt(view, event) };
      window.measuredView = view;
      window.measuring = view;
      drawMeasure(view);
    });
  };
  document.addEventListener("mousemove", (event) => {
    const view = window.measuring;
    if (!view) return;
    view.measure.b = timeAt(view, event);
    drawMeasure(view);
  });
  document.addEventListener("copy", (event) => {
    const view = window.measuredView;
    if (!view?.measure || !view.scroller.clientWidth || window.measuring || !event.clipboardData) return;
    if (!window.getSelection()?.isCollapsed || event.target.isContentEditable || event.target.closest?.("textarea, input:not([type=radio]):not([type=checkbox])")) return;
    const m = view.measure;
    const heading = view.closest(".model-panel")?.querySelector("h1")?.textContent || document.title;
    const rows = view.selectedKernels.map((el) => JSON.parse(el.dataset.kernel));
    const columns = ["id", "device", "stream", "start_us", "duration_us", "start_ns", "name"];
    const windowLine = `# ${heading}: ${Math.min(m.a, m.b).toFixed(3)}–${Math.max(m.a, m.b).toFixed(3)} µs`;
    const lines = rows.map((row) => columns.map((key) => row[key]).join("\t"));
    event.clipboardData.setData("text/plain", [windowLine, columns.join("\t"), ...lines].join("\n"));
    event.preventDefault();
  });
  document.addEventListener("mouseup", () => {
    const view = window.measuring;
    if (!view) return;
    window.measuring = null;
    if (Math.abs(view.measure.b - view.measure.a) * view.scale < 3) view.measure = null;
    drawMeasure(view);
  });
  // Labels stick to the visible part of their box, and show only when they fit in it (~5.6 px per char).
  const placeLabels = (view) => {
    const left = view.scroller.scrollLeft;
    const right = left + view.scroller.clientWidth;
    for (const el of view.labels) {
      const a = Math.max(+el.dataset.a * view.scale, left);
      const b = Math.min(+el.dataset.b * view.scale, right);
      el.setAttribute("x", a + 2);
      el.style.display = b - a > el.textContent.length * 5.6 + 4 ? "" : "none";
    }
  };
  const drawAxis = (view, svg, scale) => {
    const axis = svg.querySelector(".axis");
    axis.replaceChildren();
    const raw = 100 / scale;  // at least 100 px between ticks
    const power = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 5, 10].map((m) => m * power).find((s) => s >= raw);
    const ns = "http://www.w3.org/2000/svg";
    for (let t = 0; t <= view.span; t += step) {
      const line = document.createElementNS(ns, "line");
      line.setAttribute("x1", t * scale); line.setAttribute("x2", t * scale);
      line.setAttribute("y1", 14); line.setAttribute("y2", svg.getAttribute("height") - 8);
      line.setAttribute("stroke", "#e7ecf3");
      const text = document.createElementNS(ns, "text");
      text.setAttribute("x", t * scale + 2); text.setAttribute("y", 11);
      text.setAttribute("font-size", 10); text.setAttribute("fill", "#475569");
      text.textContent = `${+t.toFixed(3)} µs`;
      axis.append(line, text);
    }
  };
  const setup = (view) => {
    const scroller = view.querySelector(".scroller");
    view.span = +view.dataset.span;
    view.scroller = scroller;
    view.boxes = [...view.querySelectorAll("rect[data-a]")];
    view.labels = [...view.querySelectorAll("text[data-a]")];
    view.kernels = [...view.querySelectorAll("[data-kernel]")].sort((a, b) => +a.dataset.a - +b.dataset.a);
    setupMeasure(view);
    scroller.addEventListener("scroll", () => view.scale && placeLabels(view));
    new ResizeObserver(() => {  // hidden tabs have no width until shown
      if (!scroller.clientWidth) return;
      const atFit = !view.fit || view.scale === view.fit;
      view.fit = (scroller.clientWidth - 10) / view.span;
      if (atFit) layout(view, view.fit);
    }).observe(scroller);
    scroller.addEventListener("wheel", (event) => {
      if (!(event.metaKey || event.ctrlKey) || !view.fit) return;  // ctrlKey: trackpad pinch
      event.preventDefault();
      const mouse = event.clientX - scroller.getBoundingClientRect().left;
      const t = (scroller.scrollLeft + mouse) / view.scale;
      const scale = Math.min(Math.max(view.scale * Math.exp(-event.deltaY * 0.002), view.fit), MAX_PX_PER_US);
      layout(view, scale);
      scroller.scrollLeft = t * scale - mouse;  // keep the time under the cursor in place
    }, { passive: false });
  };
  const setupAll = () => document.querySelectorAll(".timeline").forEach(setup);
  // Some viewers run the script after parsing finished, then DOMContentLoaded never fires again.
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", setupAll); else setupAll();
  const tip = document.createElement("div");
  tip.className = "detail-tip";
  tip.hidden = true;
  document.body.appendChild(tip);
  window.detailTip = tip;
  document.addEventListener("mousemove", (event) => {
    const hit = !window.measuring && event.target.closest && event.target.closest("[data-detail]");
    tip.hidden = !hit;
    if (!hit) return;
    tip.textContent = hit.dataset.detail;
    const left = Math.min(event.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    const top = event.clientY + 16 + tip.offsetHeight > window.innerHeight ? event.clientY - tip.offsetHeight - 12 : event.clientY + 16;
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  });
}
</script>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, help="results/<RUN_ID>, the fetched run")
    parser.add_argument("report", type=Path, help="report/<RUN_ID>, with sol.py outputs per c<N>/")
    args = parser.parse_args()
    levels = sorted((d for d in args.report.glob("c*") if (d / "ops.parquet").exists()), key=lambda d: int(d.name[1:]))
    run = run_info(args.results)
    pages = [(d.name, d.name, page(d, args.results / d.name, run)) for d in levels]
    output = args.report / "index.html"
    output.write_text(model_tabs(pages, title=f"{run['name']}, spec-decode SOL"))
    print(output)


def run_info(results: Path) -> dict:
    """The workload name from the run's config, and the vLLM commit the server imported."""
    config = results / "config.yaml"
    name = yaml.safe_load(config.read_text()).get("name", results.name) if config.exists() else results.name
    commit = (results / "source-commit.txt").read_text().split(maxsplit=2) + ["", ""]
    return {"name": name, "commit": commit[0][:10], "date": commit[1]}


def page(report_dir: Path, run_dir: Path, run: dict) -> str:
    ops = pd.read_parquet(report_dir / "ops.parquet")
    ops["phase"] = ops.phase.map(method_path)
    ranges = pd.read_parquet(report_dir / "ranges.parquet")
    ranges["label"] = ranges.label.map(method_path)
    steps = pd.read_parquet(report_dir / "steps.parquet")
    meta = json.loads((report_dir / "meta.json").read_text())
    steady = steps[steps.steady]
    steady_ops = ops[ops.step.isin(steady.step)]
    n = len(steady)
    shown_step = meta["shown_steps"][len(meta["shown_steps"]) // 2]
    has_sol = has_kernel_sol(steady_ops)
    top = breakdown_table(steady_ops, ["phase", "name"], n)
    top = top if has_sol else top.sort_values("cost", ascending=False)
    sections = [
        f"<h1>{html.escape(run['name'])}, {report_dir.name}</h1>",
        '<p class="metric-note">NSYS shape diagnostic. Annotation overhead is present, use the normal run for performance comparisons.</p>' if meta.get("shape_diagnostic") else "",
        "" if has_sol else NO_KERNEL_SOL_NOTE,
        "<h2>Timeline</h2>",
        timeline(ops[ops.step == shown_step], ranges, steps.set_index("step").loc[shown_step]),
        "<h2>Performance</h2>",
        performance(steady_ops, steady),
        "<h2>By CPU phase of the launch</h2>",
        table_html(breakdown_table(steady_ops, "phase", n), has_sol),
        "<h2>By kernel category</h2>",
        table_html(breakdown_table(steady_ops, "category", n), has_sol),
        f"<h2>Top kernels by {'lost time' if has_sol else 'cost'}</h2>",
        table_html(top.head(25), has_sol),
        "<h2>Overview</h2>",
        overview(run_dir, steady, steady_ops, meta, run),
        footer(),
    ]
    body = "\n".join(sections)
    return f"<html><head><style>{THEME}{STYLE}</style></head><body>{body}</body></html>"


# --- Sections ---------------------------------------------------------------------------------------------------


def method_path(phase: str) -> str:
    """'GPUModelRunner.sample_tokens>DSparkSpeculator.propose' -> 'GPUModelRunner.sample_tokens() > DSparkSpeculator.propose()'."""
    labels = [LEGACY_LABELS.get(label, label) for label in phase.split(">")]
    return " > ".join(f"{label}()" if "." in label else label for label in labels)


LEGACY_LABELS = {  # short labels of the first probe version (runs from 2026-11-02)
    "execute": "GPUModelRunner.execute_model", "prepare": "GPUModelRunner.prepare_inputs",
    "sample_tokens": "GPUModelRunner.sample_tokens", "sample": "GPUModelRunner.sample",
    "postprocess": "GPUModelRunner.postprocess_sampled", "graph": "CudaGraphManager.run_fullgraph",
    "piecewise": "CudaGraphManager.run_pw_graph", "draft": "AutoRegressiveSpeculator.propose",
}  # fmt: skip


def overview(run_dir: Path, steady: pd.DataFrame, steady_ops: pd.DataFrame, meta: dict, run: dict) -> str:
    hw = meta["hw"]
    command = json.loads((run_dir / "nsys/command.json").read_text())
    server = command[command.index("-m") - 1 :]
    total = steady.cost_ns.mean() / 1e3
    median = steady.cost_ns.median() / 1e3
    floor = sol_floor(steady_ops, steady)
    reference_run = reference(run_dir / "nsys/aiperf-reference")
    if meta.get("shape_diagnostic"):
        reference_run = {"text": "Shape diagnostic: graph construction was captured for metadata. Annotation overhead is present, the unprofiled reference run was omitted."}
    items = [
        batch_line(steady),
        step_cost_line(total, median, floor),
        *stall_item(steady, floor, host_gaps(steady_ops)),
        analytic_line(steady),
        reference_run["text"],
        *speculation_items(steady, reference_run, total),
        f"GPU: {html.escape(hw['gpu'])}, peak DRAM BW {hw['bw'] / 1e12:.2f} TB/s",
        f"vLLM: {html.escape(run['commit'])} ({html.escape(run['date'])})",
        "Checks: " + ", ".join(f"{k}: {v}" for k, v in meta["checks"].items()),
    ]
    if "analytic_assumptions" in steady:
        assumptions = sorted({item for value in steady.analytic_assumptions.dropna() for item in json.loads(value)})
        items.append("Estimate assumptions: " + ". ".join(assumptions))
    items = "".join(f"<li>{item}</li>" for item in items)
    command_html = html.escape(shell_lines(server))
    return f'<div class="card"><ul>{items}</ul><details><summary>vllm serve command</summary><pre>{command_html}</pre></details></div>'


def shell_lines(command: list[str]) -> str:
    """Copy-pasteable command with one flag (and its value) per line."""
    lines = [shlex.quote(command[0])]
    for arg in command[1:]:
        if arg.startswith("--"):
            lines.append(arg)
        else:
            lines[-1] += " " + shlex.quote(arg)
    return " \\\n    ".join(lines)


def analytic_line(steady: pd.DataFrame) -> str:
    if "analytic_sol_ns" not in steady:
        return "Analytic SOL: no model info in this run"
    target = steady.target_sol_ns.mean() / 1e3
    draft = steady.draft_sol_ns.mean() / 1e3
    return f"Analytic SOL: target {target:.1f} µs, draft {draft:.1f} µs ({steady.draft_forwards.mean():.1f} draft forwards per step)"


def reference(artifacts: Path) -> dict:
    """The AIPerf run before the capture. Acceptance length only when the server exports spec-decode counters."""
    exports = list(artifacts.rglob("profile_export_aiperf.json"))
    if not exports:
        return {"text": "Reference run without capture: missing"}
    export = json.loads(exports[0].read_text())
    text = f"Reference run without capture: TPOT {export['inter_token_latency']['avg']:.3f} ms, "
    text += f"{export['output_token_throughput']['avg']:.0f} output tok/s"
    result = {"text": text, "tpot_ms": float(export["inter_token_latency"]["avg"])}
    metrics_files = list(artifacts.rglob("server_metrics_export.json"))
    metrics = json.loads(metrics_files[0].read_text())["metrics"] if metrics_files else {}
    if "vllm:spec_decode_num_drafts" not in metrics:
        return result
    accepted = server_total(metrics, "vllm:spec_decode_num_accepted_tokens")
    drafts = server_total(metrics, "vllm:spec_decode_num_drafts")
    result["acceptance"] = 1 + accepted / drafts
    result["text"] = text + f", acceptance length {result['acceptance']:.2f}"
    return result


def batch_line(steady: pd.DataFrame) -> str:
    """The batch shape. `tokens` is the scheduled verify batch, not the tokens the step generates."""
    per_request = steady.tokens.sum() / steady.reqs.sum()
    return (f"Steady steps: {len(steady)}, on average {steady.reqs.mean():.1f} requests x {per_request:.1f} scheduled tokens per request "
            f"({per_request - 1:.1f} drafts + 1 verify), {steady.context.mean() / steady.reqs.mean():.0f} context tokens per request")


def step_cost_line(total: float, median: float, floor: tuple[str, float] | None) -> str:
    text = f"Mean step cost: {total:.1f} µs (median {median:.1f} µs)"
    if not floor:
        return text
    return text + f", <b>{total / floor[1]:.2f}x {floor[0]}</b> ({floor[1]:.1f} µs), median step {median / floor[1]:.2f}x"


def stall_item(steady: pd.DataFrame, floor: tuple[str, float] | None, host: pd.Series) -> list[str]:
    """The all-steps mean and the median step disagree when a few steps stall, so say how many and how much."""
    stall = stall_summary(steady, host)
    if not stall:
        return []
    text = (f"Stalls: <b>{stall['count']} of {stall['steps']}</b> steady steps leave the GPU idle above {stall['threshold_us'] / 1e3:g} ms on the host "
            f"and hold <b>{stall['time_share']:.1%}</b> of the measured time, where the pipeline, not the kernels, sets the step cost")
    if floor and not pd.isna(stall["mean_cost_ns"]):
        text += f"; without them the mean step is {stall['mean_cost_ns'] / 1e3:.1f} µs ({stall['mean_cost_ns'] / 1e3 / floor[1]:.2f}x {floor[0]})"
    return [text]


def speculation_items(steady: pd.DataFrame, reference_run: dict, total: float) -> list[str]:
    """Scheduled drafts, generated tokens per step and the per-token cost, so the token fields cannot be misread."""
    budgets = (steady.tokens - steady.reqs) / steady.reqs
    if not budgets.max():
        return []
    constant = budgets.nunique() == 1
    budget = f"unchanged in all {len(steady)} steady steps" if constant else f"from {budgets.min():.1f} to {budgets.max():.1f} across the steady steps"
    acceptance = reference_run.get("acceptance")
    if acceptance is None:
        return [f"Speculation: {budgets.mean():.1f} drafted tokens per request, budget {budget}"]
    text = (f"Speculation: {budgets.mean():.1f} drafted tokens per request, budget {budget}, and the reference run measured an acceptance length "
            f"of {acceptance:.2f}, so one step generates {acceptance:.2f} tokens")  # fmt: skip
    if constant and "av_enabled" in steady and steady.av_enabled.all():
        text += "; adaptive verification kept the full draft budget in every steady step"
    items = [text]
    if "tpot_ms" in reference_run:
        per_token_ms = total / acceptance / 1e3
        delta = per_token_ms / reference_run["tpot_ms"] - 1
        items.append(f"Per-token cost: {total:.1f} µs/step over {acceptance:.2f} generated tokens is {per_token_ms:.3f} ms/token, "
                     f"{delta:+.1%} against the reference TPOT of {reference_run['tpot_ms']:.3f} ms")  # fmt: skip
    return items


def performance(ops: pd.DataFrame, steps: pd.DataFrame) -> str:
    """Stacked bar of the mean step cost, then the same components as a table."""
    rows = waterfall_rows(ops, steps)
    total = steps.cost_ns.mean() / 1e3
    x = 0.0
    bars = []
    for label, value in rows:
        color = component_color(label)
        share = max(value, 0) / total
        side = "right" if x > 0.5 else "left"  # keep the tooltip inside the card
        tip = html.escape(f"{label}\n{value:.1f} µs/step\n{value / total:.1%} of the step")
        bars.append(f'<div class="segment {side}" style="width:{share:.4%};background:{color}" data-tip="{tip}"></div>')
        x += share
    svg = f'<div class="stacked-bar">{"".join(bars)}</div>'
    legend = "".join(f'<span style="--c:{component_color(label)}">{html.escape(label)}</span>' for label, _ in rows)
    lines = "".join(f'<tr><td style="text-align:left">{html.escape(label)}</td><td>{value:.1f}</td><td>{value / total:.1%}</td></tr>' for label, value in rows)
    lines += f'<tr><td style="text-align:left"><b>Total step cost</b></td><td><b>{total:.1f}</b></td><td></td></tr>'
    table = f'<div class="table-scroll"><table><thead><tr><th>Component</th><th>µs/step</th><th>share</th></tr></thead><tbody>{lines}</tbody></table></div>'
    return f'<div class="card">{svg}<div class="legend">{legend}</div>{table}{performance_note(ops, steps)}</div>'


def performance_note(ops: pd.DataFrame, steps: pd.DataFrame) -> str:
    """Says which step the waterfall describes when the mean and the median disagree."""
    stall = stall_summary(steps, host_gaps(ops))
    if not stall:
        return ""
    return (f'<p class="metric-note">Components are means over all {len(steps)} steady steps. {stall["count"]} of them stall with host idle above '
            f'{stall["threshold_us"] / 1e3:g} ms and hold {stall["time_share"]:.1%} of the measured time, all of it in Idle: host. '
            f'The timeline above shows the median step instead.</p>')  # fmt: skip


def timeline(ops: pd.DataFrame, ranges: pd.DataFrame, step: pd.Series) -> str:
    """One step: the CPU calls of its interval (execute_model() up to the next one) and the GPU ops they launched.
    With async scheduling the GPU work trails the CPU, the axis spans both. Kernels are colored by x SOL, the dark
    inset is their SOL time.

    Elements carry their interval in µs from the start of the window (data-a, data-b). The page script lays them out
    at the current zoom, starting fit to width.
    """
    ops = ops if has_kernel_sol(ops) else ops.assign(sol_ns=math.nan)  # copies alone would get colors
    ranges = ranges[(ranges.start >= step.cpu_start) & (ranges.start < step.cpu_end)]
    t0 = min(ops.start.min(), step.cpu_start)
    t1 = max(ops["end"].max(), ranges["end"].max())
    streams = ops.groupby("streamId").busy_ns.sum().sort_values(ascending=False).index.tolist()
    depths = nesting(ranges)
    cpu_lanes = max(depths, default=-1) + 1
    lanes = [f"CPU depth {d}" for d in range(cpu_lanes)] + [f"stream {s}" for s in streams] + ["GPU idle"]
    lane_y = {name: 22 + i * (LANE + GAP) for i, name in enumerate(lanes)}
    us = lambda t: (t - t0) / 1e3  # noqa: E731
    shapes = []
    for r, depth in zip(ranges.itertuples(), depths):
        tip = f"CPU phase: {r.label}\nstart: {us(r.start):+.1f} µs\nduration: {(r.end - r.start) / 1e3:.1f} µs"
        shapes.append(box(us(max(r.start, t0)), lane_y[f"CPU depth {depth}"], us(min(r.end, t1)), "#cbd5e1", tip, r.label))
    for op in ops.itertuples():
        shapes.append(kernel_box(op, us, lane_y[f"stream {op.streamId}"]))
        if op.gap_ns > 0 and op.prev_end >= t0:
            tip = (f"phase: {op.phase}\nGPU idle before: {op.name}\nkind: {op.gap_kind}\n"
                   f"duration: {op.gap_ns / 1e3:.1f} µs\nhost-bound: {op.host_gap_ns / 1e3:.1f} µs")  # fmt: skip
            shapes.append(box(us(op.prev_end), lane_y["GPU idle"], us(op.start), "#9ca3af", tip))
    height = len(lanes) * (LANE + GAP) + 30
    labels = "".join(f'<text x="4" y="{y + LANE // 2 + 4}" font-size="11">{name}</text>' for name, y in lane_y.items())
    span = us(t1)
    plot = f'<svg class="plot" width="{span + 10:.0f}" height="{height}"><g class="axis"></g>{"".join(shapes)}</svg>'
    lanes_svg = f'<svg class="lanes" width="90" height="{height}">{labels}</svg>'
    view = f'<div class="timeline" data-span="{span:.1f}">{lanes_svg}<div class="scroller">{plot}</div></div>'
    legend = "".join(f'<span style="--c:{sol_color(x)}">{x}x SOL</span>' for x in [1, 1.5, 2, 4, 8])
    legend = legend if has_kernel_sol(ops) else '<span style="--c:#d1d5db">kernel (no SOL estimate)</span>'
    note = ("<p class=\"metric-note\">One steady step with the median cost. Cmd + scroll to zoom, scroll sideways to pan, drag to measure time and highlight overlapping GPU kernels, Cmd+C to copy their list, hover a box for details. The CPU lanes "
            "show this step's calls, from execute_model() to the next one. Its GPU work starts later, because with async scheduling the GPU is still busy with the previous step.</p>")  # fmt: skip
    legend += '<span style="--c:#9ca3af">GPU idle</span>'
    return f'{view}{TIMELINE_SCRIPT}<div class="legend">{legend}</div>{note}'


def table_html(table: pd.DataFrame, has_sol: bool) -> str:
    """Without ncu data, the SOL columns are dropped and rows are sorted by cost."""
    keys = list(table.index.names)
    columns = ["ops/step", "cost µs", "busy µs", "SOL µs", "idle µs", "lost µs", "x SOL"]
    if not has_sol:
        table = table.sort_values("cost", ascending=False)
        columns = ["ops/step", "cost µs", "busy µs", "idle µs"]
    head = "".join(f'<th class="name">{k}</th>' for k in keys) + "".join(f"<th>{c}</th>" for c in columns)
    rows = []
    for key, row in table.iterrows():
        cells = zip(keys, key if isinstance(key, tuple) else (key,))
        names = "".join(f'<td class="name {column}">{html.escape(str(k)).replace(" &gt; ", " &gt;<br>")}</td>' for column, k in cells)
        values = "".join(f"<td>{row[c]:.1f}</td>" if pd.notna(row[c]) else na_cell(row.mapped) for c in (["ops", "cost", "busy", "sol", "idle"] if has_sol else ["ops", "cost", "busy", "idle"]))
        if has_sol:
            x_sol = row.x_sol if pd.isna(row.x_sol) else min(row.x_sol, 99)  # >99x only for tiny copies, NA stays NA
            values += f'<td style="background:{lost_color(row.lost / table.lost.max())}">{row.lost:.1f}</td>' if pd.notna(row.lost) else na_cell(row.mapped)
            values += f'<td style="background:{sol_color(x_sol)}">{x_sol:.1f}</td>' if pd.notna(x_sol) else na_cell(row.mapped)
        rows.append(f"<tr>{names}{values}</tr>")
    return f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def na_cell(mapped: float) -> str:
    """A group without a complete SOL estimate says how much of it is mapped, so n/a cannot be read as unmapped."""
    share = math.floor(mapped * 1000) / 10  # never round a partial group up to 100%
    return f'<td title="{share:.1f}% of the busy time in this group has a SOL estimate">n/a</td>'


def footer() -> str:
    return (
        "<footer><p>Cost of an op = its busy time plus the GPU idle gap before it, so op costs sum to the step time. "
        "Kernel SOL comes from semantic work with declared shapes, precision ceilings and traffic assumptions. "
        "When semantic metadata is missing, NCU resource-demand references supply a fallback, with exact and transferred matches labeled. "
        "These describe measured implementation work, rather than a semantic lower bound. Operations missing both sources are gray. "
        "Incomplete table groups show n/a. "
        "D2D copies count a payload read and write, host transfers require a link profile. "
        "The forward reference excludes intermediate materialization and checkpoint writes. Draft query work is excluded when its shape is unknown. "
        "These optimistic roofline references omit finite-workload latency and cache effects. "
        "Timeline x SOL = kernel duration / SOL. Table x SOL = attributed cost / SOL, with overlapping busy time counted once. "
        "Idle kinds: host (launch call returned after the GPU went idle), in_graph (between nodes "
        "of one graph), launch_latency (op was queued and still waited). Graph kernels without a CUPTI launch record are assigned to "
        "the latest graph launch that has no kernels of its own, so they get its phase.</p></footer>"
    )


# --- SVG helpers ------------------------------------------------------------------------------------------------


def kernel_box(op, us, y: float) -> str:
    a = us(op.start)
    b = us(op.end)
    has_sol = op.sol_ns > 0  # NaN compares False
    x_sol = op.dur_ns / op.sol_ns if has_sol else math.nan
    kernel_id = f"step{op.step}-op{op.Index}"
    usage = ""
    if pd.notna(op.sol_pct):
        busiest = "bandwidth" if pd.notna(op.mem_pct) and op.mem_pct >= op.sm_pct else "compute"
        usage = f"ncu {busiest} usage: {op.sol_pct:.0f}% of peak"
    lines = [
        f"phase: {op.phase}",
        f"name: {op.name}",
        f"operator: {op.operator_name}" if pd.notna(getattr(op, "operator_name", None)) else "",
        f"input shapes: {op.operator_input_shapes}" if pd.notna(getattr(op, "operator_input_shapes", None)) else "",
        f"shape estimate: {op.operator_estimate_reason}" if pd.notna(getattr(op, "operator_estimate_reason", None)) and op.operator_estimate_reason else "",
        f"duration: {op.dur_ns / 1e3:.3f} µs",
        f"category: {op.category}",
        f"busy: {op.busy_ns / 1e3:.1f} µs",
        f"bandwidth SOL: {op.sol_memory_ns / 1e3:.3f} µs" if pd.notna(getattr(op, "sol_memory_ns", None)) else "",
        f"compute SOL: {op.sol_compute_ns / 1e3:.3f} µs" if pd.notna(getattr(op, "sol_compute_ns", None)) else "",
        f"SOL: {op.sol_ns / 1e3:.1f} µs, from {op.sol_source}" if has_sol else "SOL: no estimate for this kernel",
        "" if not has_sol else f"duration is {x_sol:.1f}x SOL" if x_sol < 100 else "duration is >99x SOL",
        f"estimate confidence: {getattr(op, 'sol_quality', getattr(op, 'model_quality', 'unsupported'))}",
        f"why: {getattr(op, 'sol_reason', getattr(op, 'model_reason', ''))}",
        "assuming: " + ", ".join(json.loads(getattr(op, "sol_assumptions", getattr(op, "model_assumptions", "[]")))),
        "launch: inferred (no CUPTI launch record)" if op.inferred_launch else "",
        usage,  # none for copies
        f"idle before launch: {op.gap_ns / 1e3:.1f} µs ({op.gap_kind})" if op.gap_ns > 0 else "",
    ]
    inset = ""
    if has_sol:
        inset = rect(a, min(b, a + op.sol_ns / 1e3), y + LANE * 0.65, LANE * 0.35, "#1f2937", ' opacity=".55"')
    attrs = ""
    if op.kind == "kernel":
        record = {"id": kernel_id, "device": op.deviceId, "stream": op.streamId, "start_us": f"{a:.3f}",
                  "duration_us": f"{op.dur_ns / 1e3:.3f}", "start_ns": str(op.start), "name": op.name}
        attrs = f' data-a="{a:.3f}" data-b="{b:.3f}" data-kernel="{html.escape(json.dumps(record))}"'
    return box(a, y, b, sol_color(x_sol) if has_sol else "#d1d5db", "\n".join(filter(None, lines)), extra=inset, attrs=attrs)


def box(a: float, y: float, b: float, color: str, tip: str, label: str = "", extra: str = "", attrs: str = "") -> str:
    """A lane element from a to b µs. The page script shows data-detail as a tooltip on hover."""
    text = ""
    if label:
        text = f'<text x="{a + 2:.2f}" y="{y + LANE // 2 + 4}" font-size="10" data-a="{a:.3f}" data-b="{b:.3f}">{html.escape(label)}</text>'
    return f'<g class="hit" data-detail="{html.escape(tip)}"{attrs}>{rect(a, b, y, LANE, color)}{extra}{text}</g>'


def rect(a: float, b: float, y: float, height: float, color: str, attrs: str = "") -> str:
    """Drawn at 1 px per µs until the page script applies the zoom."""
    return (f'<rect x="{a:.3f}" y="{y:.1f}" width="{max(b - a, 0.6):.3f}" height="{height:.1f}" fill="{color}" '
            f'data-a="{a:.3f}" data-b="{b:.3f}"{attrs}/>')  # fmt: skip


def nesting(ranges: pd.DataFrame) -> list[int]:
    """Depth of each range in its push/pop stack, ranges sorted by start."""
    ends: list[int] = []
    depths = []
    for r in ranges.itertuples():
        while ends and ends[-1] <= r.start:
            ends.pop()
        depths.append(len(ends))
        ends.append(r.end)
    return depths


def component_color(label: str) -> str:
    return next((color for prefix, color in COMPONENT_COLORS.items() if label.startswith(prefix)), "#d8b4b4")


def lost_color(fraction: float) -> str:
    """Red with opacity proportional to the lost time, as a fraction of the largest in the table."""
    return f"rgba(220, 38, 38, {0.85 * max(fraction, 0):.2f})"


def sol_color(x_sol: float) -> str:
    """Green at SOL, yellow at 2x, red from 8x, on a log scale."""
    t = min(max(math.log2(x_sol) / 3, 0), 1) if x_sol > 0 else 1
    return f"hsl({130 * (1 - t):.0f}, 70%, {62 - 10 * t:.0f}%)"


if __name__ == "__main__":
    main()
