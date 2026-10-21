"""Check full-domain pricing, frozen provenance and unchanged acceptance scores."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from av_full_curve import FullCurveProbe, freeze_curves, validate_curves


class FullCurveTests(unittest.TestCase):
    def test_prefill_without_budget_decision_still_starts_calibration(self) -> None:
        probe = FullCurveProbe.__new__(FullCurveProbe)
        probe.policy = {}
        probe.refresh = lambda: setattr(probe, "policy", dict(mode="calibrate", phase="tail"))
        probe.drain = lambda: None
        batch = SimpleNamespace(num_reqs=1, num_tokens=16377, num_tokens_after_padding=16377,
                                is_prefilling_np=np.array([True]))
        cuda = SimpleNamespace(Event=lambda **kwargs: SimpleNamespace(record=lambda: None))
        with patch.dict("sys.modules", torch=SimpleNamespace(cuda=cuda)):
            probe.before_forward(batch, SimpleNamespace(cg_mode="NONE"), False)
        self.assertEqual(probe.batch["phase"], "tail")
        self.assertEqual(probe.batch["prefill_requests"], 1)
        self.assertEqual(len(probe.events), 4)

    def test_freeze_requires_measured_endpoints_and_never_extrapolates(self) -> None:
        rows = [dict(num_reqs=r, num_tokens=t, drafter_ms=d, forward_ms=v)
                for r, t, d, v in ((1, 1, 1, 2), (128, 1024, 3, 6), (256, 16377, 4, 20))
                for _ in range(3)]
        curves = freeze_curves(rows, [1, 512, 1024, 6144])
        validate_curves(curves)
        self.assertEqual(curves["verify_curve"][-1], [16377.0, 20.0])
        with self.assertRaises(ValueError):
            freeze_curves(rows[:-3], [1, 512, 1024, 6144])

    def test_mixed_and_unmeasured_request_counts_always_use_frozen_table(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "curves.json"
            path.write_text(json.dumps(dict(draft_curve=[[1, 1], [256, 4]],
                                            verify_curve=[[1, 2], [16384, 20]])))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manager = SimpleNamespace(req_states=SimpleNamespace(max_num_reqs=256, max_num_batched_tokens=16384))
            manager.cost_tables = (np.full(257, 10.), np.full(16385, 20.))
            manager.set_cost_curves = lambda *args: setattr(manager, "cost_tables", (np.ones(257), np.arange(16385) + 2))
            probe = FullCurveProbe.__new__(FullCurveProbe)
            probe.refresh = lambda: None
            probe.policy = dict(mode="measured", phase="test", curve=str(path), sha256=digest)
            probe.curve_path = None
            probe.native_tables = None
            probe.counts = {"decisions": 0, "measured_decisions": 0, "mixed_decisions": 0}
            prefix = np.array([253., 254., 254.5])
            values = dict(self=manager, req_ids=list(range(253)), num_non_draft_tokens_total=1370,
                          max_draft_budget=2, num_non_draft_tokens=np.array([1118, *([1] * 252)]),
                          costs=np.array([999., 999., 999.]), num_tokens_to_estimated_accepted_tokens=prefix)
            np.testing.assert_array_equal(probe.get_costs(values), [1373, 1374, 1375])
            np.testing.assert_array_equal(prefix, [253., 254., 254.5])
            self.assertEqual(probe.counts["mixed_decisions"], 1)
            measured_tables = manager.cost_tables
            probe.policy["mode"] = "native"
            probe.counts["native_decisions"] = 0
            np.testing.assert_array_equal(probe.get_costs(values), [30, 30, 30])
            self.assertIs(manager.cost_tables, probe.native_tables)
            probe.policy["mode"] = "measured"
            np.testing.assert_array_equal(probe.get_costs(values), [1373, 1374, 1375])
            self.assertIs(manager.cost_tables, measured_tables)
            values["num_non_draft_tokens_total"] = 16384
            with self.assertRaises(RuntimeError):
                probe.get_costs(values)


if __name__ == "__main__":
    unittest.main()
