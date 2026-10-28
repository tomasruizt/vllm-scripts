"""Collect every AIPerf trial into one long-format metrics.json (one row per variant, concurrency and trial)."""
import json
from pathlib import Path
import re
import sys

REPLAY_ON = r"Decoder SWA bounded replay: in eager prefill steps, layer (\d+) .*layers (\d+)-(\d+)"
REPLAY_OFF = r"Decoder SWA bounded replay is off: (.*)\."


def main() -> None:
    root = Path(sys.argv[1])
    rows = [load_trial(path, root) for path in sorted(root.glob("*/c*/trial-*/artifacts/profile_export_aiperf.json"))]
    (root / "metrics.json").write_text(json.dumps(rows) + "\n")
    print(f"Collected {len(rows)} trials, {sum(row['requests'] for row in rows)} completed requests")


def load_trial(path: Path, root: Path) -> dict:
    variant, cell, trial = path.relative_to(root).parts[:3]
    summary = json.loads(path.read_text())
    metrics = json.loads(path.with_name("server_metrics_export.json").read_text())["metrics"]
    log = (root / variant / "server.log").read_text()
    stat = lambda key, field="p50": summary[key][field]
    duration = stat("benchmark_duration", "avg")
    drafts, accepted = total(metrics, "vllm:spec_decode_num_drafts"), total(metrics, "vllm:spec_decode_num_accepted_tokens")
    replay_on, replay_off = re.findall(REPLAY_ON, log), re.findall(REPLAY_OFF, log)
    return dict(variant=variant, concurrency=int(cell.removeprefix("c")), trial=int(trial.removeprefix("trial-")),
                requests=int(stat("request_count", "avg")), error_rate=stat("request_error_rate", "avg"),
                ttft_p50_ms=stat("time_to_first_token"), ttft_avg_ms=stat("time_to_first_token", "avg"),
                tpot_p50_ms=stat("inter_token_latency"), e2e_p50_ms=stat("request_latency"),
                e2e_avg_ms=stat("request_latency", "avg"), isl_avg=stat("input_sequence_length", "avg"),
                osl_avg=stat("output_sequence_length", "avg"),
                output_tok_s=stat("output_token_throughput", "avg"),
                total_tok_s=(stat("total_isl", "avg") + stat("total_osl", "avg")) / duration,
                acceptance_length=1 + accepted / drafts if drafts else None,
                preemptions=total(metrics, "vllm:num_preemptions"),
                replay=("on" if replay_on else "off" if replay_off else "unknown"),
                replay_detail=json.dumps(replay_on[-1] if replay_on else replay_off[-1] if replay_off else None))


def total(metrics: dict, name: str) -> float:
    return sum(series["stats"]["total"] for series in metrics.get(name, {}).get("series", []))


if __name__ == "__main__":
    main()
