"""Exercise the live-server transition from exploration to frozen-cost decisions."""

import json
import unittest

import numpy as np

from av_cost_experiment import ExperimentProbe
import test_cost_oracle_policy as policy_tests
from test_cost_oracle_policy import policy_values


class ExperimentProbeTests(policy_tests.CostPolicyTests):
    def setUp(self) -> None:
        super().setUp()
        self.probe.handle.close()
        self.probe = ExperimentProbe(7)

    def test_sweep_then_frozen_lookup_keeps_scores(self) -> None:
        values = policy_values(1)
        prefix = values["num_tokens_to_estimated_accepted_tokens"].copy()
        self.control.write_text(json.dumps(dict(mode="sweep", phase="burst-c1")))
        self.probe.next_refresh = 0
        budgets = []
        for _ in range(8):
            self.probe.get_costs(values)
            budgets.append(self.probe.select(values))
        self.assertEqual(sorted(budgets), list(range(8)))
        lookup = self.root / "frozen.json"
        rows = [dict(samples=5, mean_ms=1.0 if b == 3 else 10.0, ms=2.0) for b in range(8)]
        lookup.write_text(json.dumps(dict(max_num_seqs=256, tensor_parallel_size=1, minimum_samples=5,
                                         capacities={"1": 7}, lookup={"pooled/1": rows})))
        self.control.write_text(json.dumps(dict(mode="oracle", phase="perf-c1", lookup=str(lookup))))
        self.probe.next_refresh = 0
        values["costs"] = self.probe.get_costs(values)
        values["draft_budget"] = int(np.argmax(prefix / values["costs"]))
        self.assertEqual(self.probe.select(values), 3)
        self.assertIsNone(self.probe.active)
        np.testing.assert_array_equal(prefix, values["num_tokens_to_estimated_accepted_tokens"])

    def test_logits_limited_candidates_are_fully_swept(self) -> None:
        values = policy_values(256)
        values["max_draft_budget"] = 825
        self.control.write_text(json.dumps(dict(mode="sweep", phase="burst-c256")))
        self.probe.next_refresh = 0
        budgets = []
        for _ in range(826):
            self.probe.get_costs(values)
            budgets.append(self.probe.select(values))
        self.assertEqual(sorted(budgets), list(range(826)))
        self.assertTrue(self.probe.pending["eligible"])
        self.assertEqual(self.probe.pending["max_total_logits"], 1081)
        values["scheduled_drafts"][0] = 6
        self.assertFalse(self.probe.eligible(values))

    def test_complete_logits_limited_curve_replaces_only_feasible_costs(self) -> None:
        from av_cost_oracle_policy import complete_tables
        rows = [dict(samples=5, mean_ms=2.0, ms=2.0) for _ in range(826)]
        self.assertEqual(complete_tables({"pooled/256": rows}, 5), {})
        self.probe.tables = complete_tables({"pooled/256": rows}, 5, {256: 825})
        values = policy_values(256)
        values["max_draft_budget"] = 825
        self.assertEqual(len(self.probe.get_costs(values)), 826)
        values["max_draft_budget"] = 824
        self.assertIs(self.probe.get_costs(values), values["costs"])

    def test_targeted_sampling_preserves_scores_and_actual_budget(self) -> None:
        values = policy_values(256)
        values["max_draft_budget"] = 825
        original = values["num_tokens_to_estimated_accepted_tokens"].copy()
        self.control.write_text(json.dumps(dict(mode="sweep", phase="burstfill-c256", budgets={"256": [312, 432, 816]})))
        self.probe.next_refresh = 0
        for _ in range(30):
            self.probe.get_costs(values)
            chosen = self.probe.select(values)
            self.assertIn(chosen, [312, 432, 816])
            self.assertEqual(self.probe.pending["budget"], chosen)
        np.testing.assert_array_equal(original, values["num_tokens_to_estimated_accepted_tokens"])

    def test_control_change_is_seen_without_refresh_timer_delay(self) -> None:
        values = policy_values(256)
        values["max_draft_budget"] = 825
        for budget in (312, 816):
            temporary = self.control.with_suffix(".tmp")
            temporary.write_text(json.dumps(dict(mode="sweep", phase=f"burstfill{budget}-c256", budgets={"256": [budget]})))
            temporary.replace(self.control)
            self.probe.get_costs(values)
            self.assertEqual(self.probe.select(values), budget)


if __name__ == "__main__":
    unittest.main()
