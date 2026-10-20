"""Check exploration coverage, native decisions and timing accounting."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from av_cost_oracle import CostProbe


class CostProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.control = self.root / "control.json"
        self.control.write_text(json.dumps(dict(mode="sweep", phase="test-c8")))
        self.environment = patch.dict(os.environ, AV_COST_CONTROL=str(self.control),
                                      AV_COST_TRACE=str(self.root / "trace.jsonl"), AV_COST_SEED="42")
        self.environment.start()
        self.probe = CostProbe(7)

    def tearDown(self) -> None:
        self.probe.handle.close()
        self.environment.stop()
        self.directory.cleanup()

    def test_every_budget_once_per_cycle_without_changing_scores(self) -> None:
        values = batch_values(8)
        original = values["num_tokens_to_estimated_accepted_tokens"].copy()
        for _ in range(2):
            budgets = [self.probe.select(values) for _ in range(57)]
            self.assertEqual(sorted(budgets), list(range(57)))
        np.testing.assert_array_equal(values["num_tokens_to_estimated_accepted_tokens"], original)

    def test_native_and_incomplete_batches_keep_original_decision(self) -> None:
        values = batch_values(8)
        values["max_draft_budget"] = 55
        self.assertEqual(self.probe.select(values), 3)
        self.control.write_text(json.dumps(dict(mode="native", phase="validation-c8")))
        self.probe.next_refresh = 0
        self.assertEqual(self.probe.select(batch_values(8)), 3)
        self.assertEqual(self.probe.pending["phase"], "validation-c8")

    def test_keeps_raw_wall_cost_separate_from_adjustment(self) -> None:
        batch = SimpleNamespace(num_reqs=1, num_tokens=4, num_draft_tokens=3,
                                num_draft_tokens_per_req=np.array([3]), num_scheduled_tokens=np.array([4]))
        with patch("av_cost_oracle.time.perf_counter_ns", side_effect=[10_000, 11_000, 100_000, 101_000]):
            self.probe.step(batch, SimpleNamespace(cg_mode="FULL"))
            self.probe.step(batch, SimpleNamespace(cg_mode="FULL"))
        self.probe.handle.flush()
        row = json.loads((self.root / "trace.jsonl").read_text())
        self.assertEqual(row["cost_ms"], 0.09)
        self.assertEqual(row["probe_ms"], 0.001)
        self.assertEqual(row["adjusted_ms"], 0.089)


def batch_values(r: int) -> dict:
    state = SimpleNamespace(num_computed_tokens_np=np.full(r, 1024, dtype=np.int32))
    return dict(req_ids=list(range(r)), max_draft_budget=7 * r, draft_budget=3,
                num_non_draft_tokens=np.ones(r, dtype=np.int32), num_non_draft_tokens_total=r,
                self=SimpleNamespace(req_states=state, _max_total_logits=1081, num_bonus_tokens=1), slots=np.arange(r),
                scheduled_drafts=np.full(r, 7, dtype=np.int32),
                num_tokens_to_estimated_accepted_tokens=np.arange(7 * r + 1, dtype=float))


if __name__ == "__main__":
    unittest.main()
