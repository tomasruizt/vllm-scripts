"""Host load samples: both log formats and the per-cell peak."""

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmarking.host_load import Sample, cell_span, peak_load, read_host_load

S = 1_000_000_000
T0 = 1_793_000_000 * S  # 2026-10-26T07:33:20Z


class HostLoadTests(unittest.TestCase):
    def test_reads_current_and_legacy_lines(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "host-load.txt"
            path.write_text("2026-10-26T07:33:20Z 12.50 9.00 8.00 3/900 123 4.5\n\n")
            self.assertEqual(read_host_load(path), [Sample(T0, 12.5, 4.5)])
            path.write_text("07:33:25 31.65 45.67 38.12 15/9025 2091\n")
            self.assertEqual(read_host_load(path, date(2026, 10, 26)), [Sample(T0 + 5 * S, 31.65, None)])
            with self.assertRaises(ValueError):
                read_host_load(path)

    def test_peak_uses_busy_cores_of_the_cell(self) -> None:
        samples = [Sample(T0 - 5 * S, 99, 90.0), Sample(T0 + 5 * S, 99, 3.0), Sample(T0 + 15 * S, 99, 70.0)]
        self.assertEqual(peak_load(samples, T0, T0 + 10 * S), 70.0)  # covers the interval ending after the cell
        self.assertEqual(peak_load(samples, T0, T0 + 5 * S), 3.0)
        self.assertIsNone(peak_load(samples, T0 + 300 * S, T0 + 310 * S))

    def test_legacy_peak_waits_for_the_load_average_lag(self) -> None:
        samples = [Sample(T0, 10.0, None), Sample(T0 + 50 * S, 80.0, None), Sample(T0 + 200 * S, 150.0, None)]
        self.assertEqual(peak_load(samples, T0, T0 + 10 * S), 80.0)

    def test_cell_span_covers_all_requests(self) -> None:
        requests = [{"metadata": {"request_start_ns": 5, "request_end_ns": 9}},
                    {"metadata": {"request_start_ns": 2, "request_end_ns": 7}}]
        self.assertEqual(cell_span(requests), (2, 9))


if __name__ == "__main__":
    unittest.main()
