"""Host CPU load during benchmark cells, from samples written by host_load.sh.

CPU contention on the benchmark host (e.g. a parallel build) slows the server's scheduling and
silently lowers throughput. `peak_load()` gives a cell's peak busy CPU cores, which
`benchmarking.aiperf.gates(max_host_load=...)` checks.
"""

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

SAMPLE_INTERVAL_S = 5  # host_load.sh default, a busy-cores sample covers the interval before it
# The 1-minute load average trails CPU demand, so contention during a cell can surface after it ends.
LOAD_AVERAGE_LAG_S = 60


@dataclass(frozen=True)
class Sample:
    time_ns: int
    load_average: float  # 1-minute
    busy_cores: float | None  # None in legacy logs without /proc/stat counters


def read_host_load(path: Path, day: date | None = None) -> list[Sample]:
    """Samples of a host_load.sh log. Legacy lines with a bare HH:MM:SS take their date from `day`."""
    samples = []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if len(fields) >= 2:
            busy = float(fields[6]) if len(fields) >= 7 else None
            samples.append(Sample(timestamp_ns(fields[0], day), float(fields[1]), busy))
    return samples


def timestamp_ns(text: str, day: date | None) -> int:
    if "T" in text:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    elif day is None:
        raise ValueError(f"Sample time {text!r} has no date, pass `day`")
    else:
        moment = datetime.combine(day, datetime.strptime(text, "%H:%M:%S").time(), timezone.utc)
    return int(moment.timestamp()) * 1_000_000_000


def peak_load(samples: list[Sample], start_ns: int, end_ns: int) -> float | None:
    """Peak busy CPU cores over the cell, None without samples.

    Legacy logs fall back to the 1-minute load average, read until its lag has passed. It also
    decays slowly after contention ends, so it can flag clean cells that follow a busy period.
    """
    if samples and all(s.busy_cores is not None for s in samples):
        values = [s.busy_cores for s in samples
                  if s.busy_cores is not None and start_ns <= s.time_ns <= end_ns + SAMPLE_INTERVAL_S * 1e9]
    else:
        values = [s.load_average for s in samples if start_ns <= s.time_ns <= end_ns + LOAD_AVERAGE_LAG_S * 1e9]
    return max(values) if values else None


def cell_span(requests: list[dict]) -> tuple[int, int]:
    """First request start and last request end of an AIPerf cell, in epoch ns."""
    return (min(r["metadata"]["request_start_ns"] for r in requests),
            max(r["metadata"]["request_end_ns"] for r in requests))
