"""Collect costs and apply a subsequently frozen lookup on the same live server."""

import json
import os
from pathlib import Path

import numpy as np

from av_cost_oracle import CostProbe
from av_cost_oracle_policy import PolicyProbe, complete_tables


class ExperimentProbe(PolicyProbe):
    def __init__(self, num_speculative_steps: int) -> None:
        self.lookup_path = None
        self.control_stamp = None
        if int(os.environ.get("TENSOR_PARALLEL_SIZE", "1")) > 1:
            from vllm.distributed import get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
            if rank:
                path = Path(os.environ["AV_COST_TRACE"])
                os.environ["AV_COST_TRACE"] = str(path.with_name(f"{path.stem}-rank{rank}{path.suffix}"))
        super().__init__(num_speculative_steps)

    def select(self, values: dict) -> int:
        if self.policy["mode"] == "sweep":
            chosen = CostProbe.select(self, values)
            r = len(values["req_ids"])
            targets = self.policy.get("budgets", {}).get(str(r))
            if self.pending["eligible"] and targets:
                chosen = int(self.rng.choice(targets))
                assert 0 <= chosen <= int(values["max_draft_budget"])
                self.pending.update(budget=chosen, sampling="targeted")
        else:
            chosen = super().select(values)
        if self.pending is not None:
            self.pending.update(max_total_logits=int(values["self"]._max_total_logits),
                                num_bonus_tokens=int(values["self"].num_bonus_tokens))
        return chosen

    def eligible(self, values: dict) -> bool:
        r = len(values["req_ids"])
        manager = values["self"]
        expected = min(7 * r, max(0, manager._max_total_logits - r * manager.num_bonus_tokens))
        return bool(np.all(values["num_non_draft_tokens"] == 1)
                    and int(values["max_draft_budget"]) == expected
                    and np.all(values["scheduled_drafts"] == 7))

    def step(self, batch: object, description: object) -> None:
        if self.policy.get("mode") == "sweep" or self.policy.get("trace", False):
            CostProbe.step(self, batch, description)

    def refresh(self) -> None:
        info = self.control.stat()
        stamp = (info.st_ino, info.st_mtime_ns, info.st_size)
        if stamp != self.control_stamp:
            self.next_refresh = 0
            self.control_stamp = stamp
        super().refresh()
        path = self.policy.get("lookup")
        if path is not None and path != self.lookup_path:
            data = json.loads(Path(path).read_text())
            assert data["max_num_seqs"] == 256
            assert data["tensor_parallel_size"] == int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
            capacities = {int(r): int(b) for r, b in data["capacities"].items()}
            self.tables = complete_tables(data["lookup"], data["minimum_samples"], capacities)
            self.lookup_path = path
