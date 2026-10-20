"""Replace only AV's cost denominator with complete directly measured curves."""

import json
import os
from pathlib import Path
import time

import numpy as np

from av_cost_oracle import CostProbe


class PolicyProbe(CostProbe):
    def __init__(self, num_speculative_steps: int) -> None:
        super().__init__(num_speculative_steps)
        lookup = json.loads(Path(os.environ["AV_COST_LOOKUP"]).read_text())["lookup"]
        minimum = int(os.environ.get("AV_COST_MIN_SAMPLES", "5"))
        self.tables = complete_tables(lookup, minimum)
        self.original_costs = None
        self.source = "native"

    def get_costs(self, values: dict) -> np.ndarray:
        self.refresh()
        self.original_costs = values["costs"]
        r = len(values["req_ids"])
        eligible = self.eligible(values)
        self.source = "native_mixed_or_incomplete"
        if eligible:
            supported = r in self.tables and len(self.tables[r]["mean_ms"]) == int(values["max_draft_budget"]) + 1
            self.source = "measured_available" if supported else "native_missing_curve"
        if self.policy["mode"] != "oracle" or self.source != "measured_available":
            return self.original_costs
        self.source = "measured"
        return self.tables[r]["mean_ms"]

    def select(self, values: dict) -> int:
        chosen = int(values["draft_budget"])
        if not self.policy.get("trace", False):
            return chosen
        super().select(values)
        prefix = values["num_tokens_to_estimated_accepted_tokens"]
        r = len(values["req_ids"])
        native = int(np.argmax(prefix / self.original_costs))
        available = self.source in ("measured", "measured_available")
        median_budget = int(np.argmax(prefix / self.tables[r]["median_ms"])) if available else None
        mean_budget = int(np.argmax(prefix / self.tables[r]["mean_ms"])) if available else None
        self.pending.update(native_budget=native, budget=chosen, cost_source=self.source,
                            req_ids=list(values["req_ids"]),
                            selected_cost_ms=float(values["costs"][chosen]),
                            measured_mean_ms=float(self.tables[r]["mean_ms"][chosen]) if available else None,
                            measured_median_ms=float(self.tables[r]["median_ms"][chosen]) if available else None,
                            mean_budget=mean_budget, median_budget=median_budget,
                            estimated_accepted_tokens=prefix.tolist(),
                            original_costs_ms=self.original_costs.tolist())
        return chosen

    def step(self, batch: object, description: object) -> None:
        if self.policy.get("trace", False):
            super().step(batch, description)

    def refresh(self) -> None:
        now = time.perf_counter_ns()
        if now < self.next_refresh:
            return
        policy = json.loads(self.control.read_text())
        if policy != self.policy:
            self.handle.flush()
            self.active = None
            self.pending = None
            self.overhead_ns = 0
        self.policy = policy
        self.next_refresh = now + 500_000_000


def complete_tables(lookup: dict, minimum: int, capacities: dict[int, int] | None = None) -> dict:
    result = {}
    for key, rows in lookup.items():
        fold, r = key.split("/")
        capacity = (capacities or {}).get(int(r), 7 * int(r))
        if fold != "pooled" or len(rows) != capacity + 1:
            continue
        if any(row is None or row["samples"] < minimum for row in rows):
            continue
        table = {name: np.array([row[field] for row in rows], dtype=np.float64)
                 for name, field in (("mean_ms", "mean_ms"), ("median_ms", "ms"))}
        if all(np.all(np.isfinite(vector) & (vector > 0)) for vector in table.values()):
            result[int(r)] = table
    return result
