"""Check window boundaries, metric scope and phase selection."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmarking.aiperf import gates, read_artifacts, server_stat, server_total, steady_window, tokens_in_window


class AIPerfTests(unittest.TestCase):
    def test_window_counts_arrivals_but_samples_latency_by_request_start(self) -> None:
        requests = [request(start) for start in (0, 2, 4)]
        result = steady_window(requests, 1)
        self.assertEqual(result, dict(window_s=1, window_requests=1, throughput=4,
                                      mean_tpot_ms=20, mean_itl_ms=1000, mean_ttft_ms=1000))

    def test_token_window_includes_start_and_excludes_end(self) -> None:
        self.assertEqual(tokens_in_window(request(0), 2e9, 3e9), 2)
        self.assertEqual(tokens_in_window(request(0), 3e9, 4e9), 2)
        self.assertEqual(tokens_in_window(request(0), 4e9, 5e9), 0)

    def test_empty_and_draining_windows_are_unusable(self) -> None:
        for requests in ([], [request(0)], [request(0), request(2)]):
            result = steady_window(requests, 2)
            self.assertEqual(result["window_s"], 0)
            self.assertIsNone(result["throughput"])
        with self.assertRaises(ValueError):
            steady_window([request(0)], 0)

    def test_server_counters_sum_but_gauges_use_maximum(self) -> None:
        metrics = {"x": {"series": [{"stats": {"total": 3, "p50": 2}}, {"stats": {"total": 7, "p50": 5}}]}}
        self.assertEqual(server_total(metrics, "x"), 10)
        self.assertEqual(server_stat(metrics, "x", "p50"), 5)
        with self.assertRaises(KeyError):
            server_total(metrics, "missing")

    def test_gate_policy_is_explicit(self) -> None:
        row = dict(concurrency=10, requests=20, requests_target=20, cancelled=0, error_rate=0,
                   preemptions=0, server_running_p50=9, server_waiting_p50=0, window_s=7)
        self.assertTrue(gates(row)["valid"])
        self.assertEqual(gates(row, min_window_s=10)["failed_gates"], "window_ge_10s")
        self.assertEqual(gates(row, saturation_ratio=1)["failed_gates"], "server_saturated")
        self.assertEqual(gates(row | {"cancelled": 1, "preemptions": 1})["failed_gates"], "all_completed,no_preemptions")

    def test_phase_selection_preserves_whole_cell_server_metrics(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "profile_export_aiperf.json"
            path.write_text(json.dumps({"input_config": {"phases": [{"name": "warmup"}, {"name": "profiling"}]}}))
            path.with_name("server_metrics_export.json").write_text('{"metrics": {"whole_cell": 100}}')
            records = [request(0) | {"metadata": {"phase_name": "warmup"}},
                       request(2) | {"metadata": {"phase_name": "profiling"}},
                       {"metadata": {"phase_name": "profiling"}, "metrics": {}}]
            path.with_name("profile_export.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n\n")
            phase_dir = path.parent / "phases/profiling"
            phase_dir.mkdir(parents=True)
            (phase_dir / path.name).write_text('{"selected": true}')
            artifacts = read_artifacts(path, phase_name="profiling")
            self.assertEqual(artifacts.summary, {"selected": True})
            self.assertEqual(artifacts.server, {"metrics": {"whole_cell": 100}})
            self.assertEqual(len(artifacts.requests), 2)
            self.assertEqual(len(artifacts.completed), 1)


def request(start_s: int) -> dict:
    return dict(metadata=dict(request_start_ns=start_s * 10**9, request_end_ns=(start_s + 3) * 10**9),
                metrics=dict(time_to_first_token={"value": 1000}, inter_token_latency={"value": 20},
                             inter_chunk_latency={"value": [1000, 1000]}, output_sequence_length={"value": 6}))
