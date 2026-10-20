"""Collect measured cost distributions, without changing AV acceptance estimates."""

import atexit
import json
import os
from pathlib import Path
import time

import numpy as np
import msgspec


class CostProbe:
    def __init__(self, num_speculative_steps: int):
        assert num_speculative_steps == 7
        self.control = Path(os.environ["AV_COST_CONTROL"])
        self.handle = Path(os.environ["AV_COST_TRACE"]).open("ab", buffering=1 << 20)
        atexit.register(self.handle.close)
        self.encoder = msgspec.json.Encoder()
        self.rng = np.random.default_rng(int(os.environ.get("AV_COST_SEED", "42")))
        self.policy = {}
        self.next_refresh = 0
        self.orders = {}
        self.pending = None
        self.active = None
        self.overhead_ns = 0
        self.sequence = int(os.environ.get("AV_COST_SEQUENCE_START", "0"))

    def select(self, values: dict) -> int:
        start = time.perf_counter_ns()
        if start >= self.next_refresh:
            policy = json.loads(self.control.read_text())
            if policy != self.policy:
                self.handle.flush()
            self.policy = policy
            self.next_refresh = start + 500_000_000
        r = len(values["req_ids"])
        capacity = int(values["max_draft_budget"])
        eligible = self.eligible(values)
        chosen = int(values["draft_budget"])
        native = chosen
        cycle = None
        if eligible and self.policy["mode"] == "sweep":
            order, offset, cycle = self.orders.get(r, (None, 0, -1))
            if order is None or offset == len(order):
                order = self.rng.permutation(capacity + 1)
                offset = 0
                cycle += 1
            chosen = int(order[offset])
            self.orders[r] = (order, offset + 1, cycle)
        contexts = values["self"].req_states.num_computed_tokens_np[values["slots"]]
        self.pending = dict(phase=self.policy["phase"], policy=self.policy["mode"],
                            eligible=eligible, native_budget=native, budget=chosen, capacity=capacity,
                            cycle=cycle, contexts=contexts.tolist(), non_draft_total=int(values["num_non_draft_tokens_total"]))
        self.overhead_ns += time.perf_counter_ns() - start
        return chosen

    def eligible(self, values: dict) -> bool:
        r = len(values["req_ids"])
        return bool(np.all(values["num_non_draft_tokens"] == 1) and int(values["max_draft_budget"]) == 7 * r)

    def step(self, batch: object, description: object) -> None:
        start = time.perf_counter_ns()
        r = int(batch.num_reqs)
        drafts = batch.num_draft_tokens_per_req
        non_draft = batch.num_scheduled_tokens - (0 if drafts is None else drafts)
        prefill = int(non_draft[non_draft > 1].sum())
        if self.active is not None:
            row = self.active
            gap = start - row.pop("start_ns")
            row.update(gap_ms=gap / 1e6, probe_ms=self.overhead_ns / 1e6,
                       cost_ms=gap / 1e6, adjusted_ms=(gap - self.overhead_ns) / 1e6,
                       next_num_reqs=r, next_prefill_tokens=prefill)
            self.handle.write(self.encoder.encode(row) + b"\n")
            if self.sequence % 256 == 0:
                self.handle.flush()
        self.sequence += 1
        self.active = dict(sequence=self.sequence, time_ns=time.time_ns(), start_ns=start,
                           num_reqs=r, num_tokens=int(batch.num_tokens),
                           num_draft_tokens=int(batch.num_draft_tokens), prefill_tokens=prefill,
                           draft_lens=[] if drafts is None else drafts[:r].tolist(),
                           cg_mode=str(description.cg_mode), **(self.pending or {}))
        self.pending = None
        self.overhead_ns = time.perf_counter_ns() - start
