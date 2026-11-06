"""AIPerf artifacts, steady-window metrics and configurable validity gates."""

from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean

import numpy as np

from benchmarking.artifacts import read_jsonl


@dataclass(frozen=True)
class Artifacts:
    summary: dict
    server: dict
    requests: list[dict]

    @property
    def completed(self) -> list[dict]:
        return [row for row in self.requests if "output_sequence_length" in row.get("metrics", {})]


def read_artifacts(path: Path, *, phase_name: str | None = None) -> Artifacts:
    """Read a summary and sibling exports, optionally selecting one AIPerf phase.

    Server metrics remain whole-cell exports. Phase filtering applies to requests
    and the summary only, not to server counter scope.
    """
    path = Path(path)
    summary = json.loads(path.read_text())
    server = json.loads(path.with_name("server_metrics_export.json").read_text())
    requests = read_jsonl(path.with_name("profile_export.jsonl"))
    if phase_name is not None:
        if len(summary["input_config"]["phases"]) > 1:
            summary = json.loads((path.parent / "phases" / phase_name / path.name).read_text())
        requests = [row for row in requests if row["metadata"]["phase_name"] == phase_name]
    return Artifacts(summary, server, requests)


def steady_window(requests: list[dict], concurrency: int) -> dict:
    """Metrics while the client holds `concurrency` requests in flight.

    The window opens at the `concurrency`-th completion (initial prefill burst is over) and closes
    when the last request is issued (drain begins). Throughput counts output tokens arriving inside it."""
    if not requests:
        return empty_window()
    if concurrency < 1:
        raise ValueError("Concurrency must be positive")
    ends = sorted(r["metadata"]["request_end_ns"] for r in requests)
    start = ends[min(concurrency, len(ends)) - 1]
    end = max(r["metadata"]["request_start_ns"] for r in requests)
    if end <= start:
        return empty_window()
    tokens = sum(tokens_in_window(r, start, end) for r in requests)
    inside = [r["metrics"] for r in requests if start <= r["metadata"]["request_start_ns"] <= end]
    tpots = [m["inter_token_latency"]["value"] for m in inside if "inter_token_latency" in m]
    itls = [v for m in inside for v in m.get("inter_chunk_latency", {}).get("value", [])]
    return dict(window_s=(end - start) / 1e9, window_requests=len(inside), throughput=float(tokens) * 1e9 / (end - start),
                mean_tpot_ms=mean(tpots) if tpots else None,
                mean_itl_ms=mean(itls) if itls else None,
                mean_ttft_ms=mean(m["time_to_first_token"]["value"] for m in inside) if inside else None)


def tokens_in_window(record: dict, start: int, end: int) -> float:
    """Output tokens of one request arriving in [start, end), spread evenly over its streamed chunks."""
    metrics = record["metrics"]
    first = record["metadata"]["request_start_ns"] + metrics["time_to_first_token"]["value"] * 1e6
    gaps = metrics.get("inter_chunk_latency", {}).get("value", [])
    times = first + 1e6 * np.cumsum([0.0, *gaps])
    per_chunk = metrics["output_sequence_length"]["value"] / len(times)
    return per_chunk * np.count_nonzero((times >= start) & (times < end))


def gates(row: dict, *, min_window_s: float = 5, saturation_ratio: float = 0.9,
          require_saturation: bool = True, max_host_load: float | None = None) -> dict:
    """A cell is valid only if it measured a saturated, unqueued, fully completed workload.

    `requests_target` is None for duration-bounded cells, where the client decides how many
    requests fit and completion is checked by `cancelled` alone. Fixed-work closed-loop cells
    drain by design, so their whole-cell running gauge is not a saturation signal.
    With `max_host_load`, the cell's `host_load_peak` (see `benchmarking.host_load`) must exist
    and stay at or below it, which rejects cells measured under CPU contention.
    """
    c, target = row["concurrency"], row["requests_target"]
    saturation = {"server_saturated": row["server_running_p50"] >= saturated_at(c, saturation_ratio)}
    host = {} if max_host_load is None else {
        "host_quiet": row["host_load_peak"] is not None and row["host_load_peak"] <= max_host_load}
    checks = dict(all_completed=(target is None or row["requests"] == target) and row["cancelled"] == 0,
                  no_errors=row["error_rate"] == 0, no_preemptions=row["preemptions"] == 0,
                  **(saturation if require_saturation else {}),
                  no_queueing=row["server_waiting_p50"] == 0, **{f"window_ge_{min_window_s:g}s": row["window_s"] >= min_window_s},
                  **host)
    failed = [name for name, ok in checks.items() if not ok]
    return dict(valid=not failed, failed_gates=",".join(failed))


def saturated_at(concurrency: int, saturation_ratio: float) -> int:
    """Server gauges are integers, so compare p50 against the nearest whole request.

    Requiring strictly `saturation_ratio * c` at c8 demands 8, one slot above what the
    gauge reports between a completion and the next send of a closed-loop client.
    """
    return max(round(saturation_ratio * concurrency), 1)


def server_total(metrics: dict, name: str) -> float:
    return sum(series["stats"]["total"] for series in metrics[name]["series"])


def server_stat(metrics: dict, name: str, stat: str) -> float:
    return max(series["stats"][stat] for series in metrics[name]["series"])


def empty_window() -> dict:
    return dict(window_s=0.0, window_requests=0, throughput=None,
                mean_tpot_ms=None, mean_itl_ms=None, mean_ttft_ms=None)
