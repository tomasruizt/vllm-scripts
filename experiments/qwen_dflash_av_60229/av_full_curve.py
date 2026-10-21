"""Real-batch calibration and unconditional measured AV cost tables."""

import atexit
from collections import defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


class FullCurveProbe:
    def __init__(self, steps: int) -> None:
        assert steps == 7
        from vllm.distributed import get_tensor_model_parallel_rank

        self.rank = get_tensor_model_parallel_rank()
        self.root = Path(os.environ["FULL_CURVE_ROOT"])
        self.control = self.root / "control.json"
        self.policy = {}
        self.stamp = None
        self.rng = np.random.default_rng(542)
        self.curve_path = None
        self.native_tables = None
        self.measured_tables = None
        self.digest = None
        self.pending = deque()
        self.events = None
        self.counts = defaultdict(int)
        self.timings = (self.root / f"timings-rank{self.rank}.jsonl").open("a", buffering=1)
        self.trace = (self.root / f"decisions-rank{self.rank}.jsonl").open("a", buffering=1)
        atexit.register(self.close)

    def get_costs(self, values: dict) -> np.ndarray:
        self.refresh()
        self.counts["decisions"] += 1
        mode = self.policy["mode"]
        if mode == "calibrate":
            return values["costs"]
        if mode not in ("native", "measured"):
            raise RuntimeError(f"Unexpected experiment mode: {mode}")
        manager = values["self"]
        if self.native_tables is None:
            self.native_tables = manager.cost_tables
        if mode == "native":
            manager.cost_tables = self.native_tables
            r = len(values["req_ids"])
            start = int(values["num_non_draft_tokens_total"])
            stop = start + int(values["max_draft_budget"]) + 1
            draft, verify = self.native_tables
            self.counts["native_decisions"] += 1
            return draft[r] + verify[start:stop]
        path = self.policy["curve"]
        if path != self.curve_path:
            raw = Path(path).read_bytes()
            data = json.loads(raw)
            validate_curves(data, manager.req_states.max_num_reqs, manager.req_states.max_num_batched_tokens - 7)
            self.digest = hashlib.sha256(raw).hexdigest()
            if self.digest != self.policy["sha256"]:
                raise RuntimeError("Frozen curve checksum mismatch")
            manager.set_cost_curves(data["draft_curve"], data["verify_curve"])
            self.measured_tables = manager.cost_tables
            self.curve_path = path
            self.max_measured_tokens = int(data["verify_curve"][-1][0])
        r = len(values["req_ids"])
        start = int(values["num_non_draft_tokens_total"])
        stop = start + int(values["max_draft_budget"]) + 1
        manager.cost_tables = self.measured_tables
        draft, verify = self.measured_tables
        if r >= len(draft) or stop > len(verify) or stop - 1 > self.max_measured_tokens:
            raise RuntimeError(f"Measured table does not cover R={r}, tokens={start}:{stop}")
        costs = draft[r] + verify[start:stop]
        if not np.all(np.isfinite(costs) & (costs > 0)):
            raise RuntimeError("Invalid measured costs")
        self.counts["measured_decisions"] += 1
        self.counts["mixed_decisions"] += int(np.any(values["num_non_draft_tokens"] > 1))
        return costs

    def select(self, values: dict) -> int:
        budget = int(values["draft_budget"])
        if self.policy["mode"] == "calibrate":
            return int(self.rng.integers(int(values["max_draft_budget"]) + 1))
        if self.policy.get("trace"):
            r = len(values["req_ids"])
            self.write(self.trace, dict(phase=self.policy["phase"], time_ns=time.time_ns(),
                requests=r, non_draft_tokens=int(values["num_non_draft_tokens_total"]),
                capacity=int(values["max_draft_budget"]), budget=budget, source=self.policy["mode"],
                sha256=self.digest, selected_ms=float(values["costs"][budget])))
        return budget

    def before_forward(self, batch: object, description: object, dummy: bool) -> None:
        # Pure prefill has no draft-budget decision, so get_costs may not run.
        if not dummy:
            self.refresh()
        self.drain()
        if dummy or self.policy.get("mode") != "calibrate":
            return
        import torch

        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        self.batch = dict(phase=self.policy["phase"], time_ns=time.time_ns(),
            num_reqs=int(batch.num_reqs), num_tokens=int(batch.num_tokens),
            padded_tokens=int(batch.num_tokens_after_padding), cg_mode=str(description.cg_mode),
            prefill_requests=int(batch.is_prefilling_np.sum()))
        self.events[0].record()

    def forward_end(self) -> None:
        if self.events is not None:
            self.events[1].record()

    def draft_start(self) -> None:
        if self.events is not None:
            self.events[2].record()

    def draft_end(self) -> None:
        if self.events is not None:
            self.events[3].record()
            self.pending.append((self.events, self.batch))
            self.events = None

    def drain(self) -> None:
        while self.pending and self.pending[0][0][3].query():
            events, row = self.pending.popleft()
            row.update(forward_ms=events[0].elapsed_time(events[1]),
                       drafter_ms=events[2].elapsed_time(events[3]))
            self.write(self.timings, row)

    def refresh(self) -> None:
        info = self.control.stat()
        stamp = (info.st_ino, info.st_mtime_ns, info.st_size)
        if stamp != self.stamp:
            self.save_audit()
            self.policy = json.loads(self.control.read_text())
            self.stamp = stamp
            self.counts.clear()

    def save_audit(self) -> None:
        if self.policy:
            path = self.root / f"audit-{self.policy['phase']}-rank{self.rank}.json"
            path.write_text(json.dumps(dict(**self.counts, mode=self.policy["mode"],
                                           sha256=self.digest if self.policy["mode"] == "measured" else None,
                                           fallback_decisions=0)))

    def close(self) -> None:
        if self.pending:
            self.pending[-1][0][3].synchronize()
            self.drain()
        self.save_audit()
        self.timings.close()
        self.trace.close()

    @staticmethod
    def write(handle: object, row: dict) -> None:
        handle.write(json.dumps(row) + "\n")


def validate_curves(data: dict, requests: int = 256, tokens: int = 16377) -> None:
    for name, limit in (("draft_curve", requests), ("verify_curve", tokens)):
        points = np.asarray(data[name], dtype=float)
        if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
                or points[0, 0] > 1 or points[-1, 0] < limit
                or np.any(np.diff(points[:, 0]) <= 0) or np.any(points[:, 1] <= 0)):
            raise ValueError(f"Measured {name} does not cover 1..{limit}")


def freeze_curves(rows: list[dict], capture_sizes: list[int], minimum: int = 3) -> dict:
    groups = {name: defaultdict(list) for name in ("draft", "verify")}
    for row in rows:
        if row.get("rank", 0) != 0:
            continue
        r = int(row["num_reqs"])
        t = int(row["num_tokens"])
        # Match the native lookup: ceil to graph buckets, interpolate above them.
        bucket = next((n for n in capture_sizes if n >= t), t)
        if row.get("cg_mode", "FULL").endswith("FULL"):
            groups["draft"][r].append(float(row["drafter_ms"]))
        groups["verify"][bucket].append(float(row["forward_ms"]))
    curves = {}
    points = {}
    # DFlash reserves k=7 extra input slots per request before target scheduling.
    # Hence even R=1 cannot query more than 16384-7 target tokens.
    for name, limit in (("draft", 256), ("verify", 16377)):
        measured = [(n, float(np.median(ms)), len(ms)) for n, ms in sorted(groups[name].items())
                    if len(ms) >= minimum and np.all(np.isfinite(ms)) and min(ms) > 0]
        if not measured or measured[0][0] != 1 or measured[-1][0] != limit:
            domain = (measured[0][0], measured[-1][0]) if measured else None
            raise ValueError(f"Insufficient measured {name} endpoints: {domain}, need 1..{limit}, refusing extrapolation")
        x, y, _ = np.asarray(measured).T
        knots = np.arange(1, 257) if name == "draft" else np.unique(
            np.concatenate((capture_sizes, x[x > max(capture_sizes)])))
        values = np.maximum.accumulate(np.interp(knots, x, y))
        curves[f"{name}_curve"] = np.column_stack((knots, values)).tolist()
        points[name] = measured
    curves.update(points=points, minimum_samples=minimum, tensor_parallel_size=2,
                  max_num_seqs=256, max_num_batched_tokens=16384, configured_drafter_k=7,
                  max_feasible_target_tokens=16377,
                  method="Real-batch CUDA event medians, interpolated missing knots, native monotone envelope and graph buckets",
                  limitations="Context/composition averaged, excludes CPU and sampling overhead, not a perfect oracle")
    validate_curves(curves)
    return curves
