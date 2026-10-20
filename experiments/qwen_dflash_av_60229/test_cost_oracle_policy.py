"""Verify denominator replacement and transparent fallback without changing acceptance scores."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from av_cost_oracle_policy import PolicyProbe, complete_tables
from test_cost_oracle import batch_values


class CostPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.control = self.root / "control.json"
        self.control.write_text(json.dumps(dict(mode="oracle", phase="test-c1", trace=True)))
        rows = [dict(samples=10, mean_ms=1.0 if b == 2 else 10.0, ms=2.0) for b in range(8)]
        self.lookup = self.root / "lookup.json"
        self.lookup.write_text(json.dumps(dict(lookup={"pooled/1": rows})))
        self.environment = patch.dict(os.environ, AV_COST_CONTROL=str(self.control),
                                      AV_COST_TRACE=str(self.root / "trace.jsonl"),
                                      AV_COST_LOOKUP=str(self.lookup), AV_COST_MIN_SAMPLES="5")
        self.environment.start()
        self.probe = PolicyProbe(7)

    def tearDown(self) -> None:
        self.probe.handle.close()
        self.environment.stop()
        self.directory.cleanup()

    def test_replaces_denominator_and_preserves_acceptance_prefix(self) -> None:
        values = policy_values(1)
        original = values["num_tokens_to_estimated_accepted_tokens"].copy()
        values["costs"] = self.probe.get_costs(values)
        chosen = int(np.argmax(original / values["costs"]))
        self.assertEqual(chosen, 2)
        values["draft_budget"] = chosen
        self.assertEqual(self.probe.select(values), chosen)
        self.assertEqual(self.probe.pending["native_budget"], 7)
        self.assertEqual(self.probe.pending["cost_source"], "measured")
        self.assertEqual(self.probe.pending["mean_budget"], 2)
        self.assertEqual(self.probe.pending["median_budget"], 7)
        self.assertEqual(self.probe.pending["selected_cost_ms"], 1.0)
        np.testing.assert_array_equal(values["num_tokens_to_estimated_accepted_tokens"], original)

    def test_native_mixed_and_missing_curves_keep_original_cost_object(self) -> None:
        values = policy_values(1)
        values["num_non_draft_tokens"][0] = 2
        self.assertIs(self.probe.get_costs(values), values["costs"])
        self.assertEqual(self.probe.source, "native_mixed_or_incomplete")
        values = policy_values(8)
        self.assertIs(self.probe.get_costs(values), values["costs"])
        self.assertEqual(self.probe.source, "native_missing_curve")
        self.control.write_text(json.dumps(dict(mode="native", phase="test-c1")))
        self.probe.next_refresh = 0
        values = policy_values(1)
        self.assertIs(self.probe.get_costs(values), values["costs"])

    def test_rejects_incomplete_under_sampled_and_nonpositive_curves(self) -> None:
        rows = [dict(samples=10, mean_ms=2.0, ms=2.0) for _ in range(8)]
        self.assertIn(1, complete_tables({"pooled/1": rows}, 5))
        for invalid in (None, dict(samples=4, mean_ms=2.0, ms=2.0), dict(samples=10, mean_ms=0.0, ms=2.0)):
            self.assertEqual(complete_tables({"pooled/1": [invalid, *rows[1:]]}, 5), {})

    def test_performance_mode_does_not_write_step_traces(self) -> None:
        self.control.write_text(json.dumps(dict(mode="oracle", phase="perf-c1", trace=False)))
        values = policy_values(1)
        values["costs"] = self.probe.get_costs(values)
        values["draft_budget"] = int(np.argmax(values["num_tokens_to_estimated_accepted_tokens"] / values["costs"]))
        self.probe.select(values)
        self.probe.step(object(), object())
        self.probe.handle.flush()
        self.assertEqual((self.root / "trace.jsonl").read_bytes(), b"")


def policy_values(r: int) -> dict:
    return dict(**batch_values(r), costs=np.full(7 * r + 1, 10.0))


if __name__ == "__main__":
    unittest.main()
