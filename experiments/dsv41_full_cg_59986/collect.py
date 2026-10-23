"""Collect all variants into one long-format metrics.json (variant, phase, concurrency, metric, value) and probes.json."""
import json
from pathlib import Path
import re
import sys

AIPERF_STATS = {"request_count": "avg", "request_error_rate": "avg", "time_to_first_token": "p50",
                "inter_token_latency": "p50", "request_latency": "p50", "output_token_throughput": "avg",
                "output_sequence_length": "avg"}


def main() -> None:
    root = Path(sys.argv[1])
    variants = sorted(path.parent for path in root.glob("*/server-command.json"))
    rows = [row for variant in variants for row in variant_rows(variant)]
    probes = [{"variant": variant.name} | probe for variant in variants
              for probe in load_json(variant / "probes/probes.json", [])]
    (root / "metrics.json").write_text(json.dumps(rows) + "\n")
    (root / "probes.json").write_text(json.dumps(probes, indent=1, ensure_ascii=False) + "\n")
    print(f"Collected {len(rows)} metric rows and {len(probes)} probes from {len(variants)} variants")


def variant_rows(variant: Path) -> list[dict]:
    row = lambda phase, metric, value, concurrency=None: dict(
        variant=variant.name, phase=phase, concurrency=concurrency, metric=metric, value=value)
    log = (variant / "server.log").read_text(errors="replace")
    info = load_json(variant / "server-info.json", {})
    rows = [row("server", "cudagraph_mode", str(find_key(info, "cudagraph_mode"))),
            row("server", "full_graphs_captured", graphs_captured(log, "FULL")),
            row("server", "piecewise_graphs_captured", graphs_captured(log, "PIECEWISE")),
            row("server", "downgrade_warning", "; ".join(sorted(set(re.findall(r"CUDAGraphMode\.\w+ is not supported[^\n]*", log))))),
            row("server", "illegal_memory_access", "illegal memory access" in log),
            row("server", "exit_code", read(variant / "exit-code.txt")),
            row("server", "alive_after_phases", read(variant / "server-alive.txt"))]
    gsm8k = load_json(variant / "gsm8k/gsm8k.json", {})
    rows += [row("gsm8k", key, gsm8k[key]) for key in ("accuracy", "invalid_rate", "total_output_tokens") if key in gsm8k]
    rows += [row("probes", f"{p['probe']}_distinct8", distinct_ratio(p)) for p in load_json(variant / "probes/probes.json", [])]
    rows += aiperf_rows(variant / "load", "load", row)
    rows += [r for cell in sorted((variant / "perf").glob("c*"))
             for r in aiperf_rows(cell, "perf", row, int(cell.name.removeprefix("c")))]
    return rows


def aiperf_rows(cell: Path, phase: str, row, concurrency: int | None = None) -> list[dict]:
    summaries = list(cell.glob("artifacts/**/profile_export_aiperf.json"))
    if not summaries:
        return [row(phase, "client_exit_code", read(cell / "client-exit-code.txt"), concurrency)] if cell.exists() else []
    summary = json.loads(summaries[0].read_text())
    rows = [row(phase, f"{key}_{field}", summary[key][field], concurrency)
            for key, field in AIPERF_STATS.items() if key in summary]
    metrics = load_json(summaries[0].with_name("server_metrics_export.json"), {}).get("metrics", {})
    drafts, accepted = total(metrics, "vllm:spec_decode_num_drafts"), total(metrics, "vllm:spec_decode_num_accepted_tokens")
    return rows + [row(phase, "acceptance_length", 1 + accepted / drafts if drafts else None, concurrency)]


def graphs_captured(log: str, mode: str) -> int:
    """Sum the totals of finished capture progress bars: a bar's last update reads N/N and tqdm may repeat it."""
    finished = (re.findall(r"\b(\d+)/\1\b", line) for line in log.splitlines() if f"Capturing CUDA graphs ({mode})" in line)
    return sum(max(map(int, totals)) for totals in finished if totals)


def distinct_ratio(probe: dict, n: int = 8) -> float | None:
    """Unique character n-grams over all n-grams of reasoning plus content: repetition loops score low."""
    text = (probe.get("reasoning") or "") + (probe.get("content") or "")
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)) / len(grams) if grams else None


def find_key(data, key: str):
    if isinstance(data, dict):
        if key in data:
            return data[key]
        data = list(data.values())
    for value in data if isinstance(data, list) else []:
        if (found := find_key(value, key)) is not None:
            return found
    return None


def total(metrics: dict, name: str) -> float:
    return sum(series["stats"]["total"] for series in metrics.get(name, {}).get("series", []))


def load_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def read(path: Path) -> str | None:
    return path.read_text().strip() if path.exists() else None


if __name__ == "__main__":
    main()
