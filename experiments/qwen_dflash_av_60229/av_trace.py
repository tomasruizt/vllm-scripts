"""Lightweight CPU-only AV diagnostics, enabled by AV_TRACE_PATH. Hooked in by av-diagnostics.patch.

Per step it records scalars only, so tracing barely perturbs step timing:
- "budget": AV's draft-budget decision (AV on only).
- "output": tokens sampled in the step (AV on and off), timestamped when outputs reach the CPU.
"""

import json
import os
import time
from pathlib import Path

import numpy as np

_handle = None


def emit(kind: str, **fields: object) -> None:
    global _handle
    path = os.environ.get("AV_TRACE_PATH")
    if not path:
        return
    if _handle is None:
        _handle = Path(path).open("a", buffering=1)
    record = dict(kind=kind, time_ns=time.time_ns(), **fields)
    _handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def costs(manager: object, draft_curve: list, verify_curve: list) -> None:
    emit("costs", draft_curve=draft_curve, verify_curve=verify_curve,
         cudagraph_limit=manager._cudagraph_limit)


def budget(v: dict) -> None:
    non_draft = v["num_non_draft_tokens"]
    est, cost = v["num_tokens_to_estimated_accepted_tokens"], v["costs"]
    b, max_b = int(v["draft_budget"]), int(v["max_draft_budget"])
    emit("budget", num_reqs=len(v["req_ids"]), non_draft_total=int(v["num_non_draft_tokens_total"]),
         has_prefill=bool((non_draft > 1).any()), scheduled_drafts=int(v["scheduled_drafts"].sum()),
         max_budget=max_b, budget=b, est_tokens=float(est[b]), est_tokens_max=float(est[max_b]),
         cost_ms=float(cost[b]), cost_ms_max=float(cost[max_b]))


def step(input_batch: object, batch_desc: object) -> None:
    """One record per executed step, with its final shape (after AV trimming). Step time = gap to the next record."""
    drafts = input_batch.num_draft_tokens_per_req
    non_draft = input_batch.num_scheduled_tokens - (0 if drafts is None else drafts)
    emit("step", num_reqs=int(input_batch.num_reqs), num_tokens=int(input_batch.num_tokens),
         num_draft_tokens=int(input_batch.num_draft_tokens), prefill_tokens=int(non_draft[non_draft > 1].sum()),
         cg_mode=str(batch_desc.cg_mode))


def output(req_ids: list, num_sampled: list) -> None:
    emit("output", num_reqs=len(req_ids), num_sampled=int(np.sum(num_sampled)))
