"""Collect native tail timings, metadata and torch-profiler kernel groups."""

import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    root = Path("results/qwen-av-tail-profile-20261019")
    metrics = []
    kernels = []
    for arm in ("slot_only", "mixed"):
        for batch in json.loads((root / arm / "batches.json").read_text()):
            validate(batch)
            metrics.extend(dict(policy=batch["policy"], tokens=batch["tokens"],
                                repeat=repeat, gpu_ms=ms, batch=batch["batch"])
                           for repeat, ms in enumerate(batch["target_ms"], 1))
            kernels.extend(kernel_rows(root / arm, batch))
    (root / "timings.json").write_text(json.dumps(metrics))
    (root / "kernels.json").write_text(json.dumps(kernels))
    print(f"Validated {len(metrics)} timings and {len(kernels)} kernel groups")


def validate(record: dict) -> None:
    batch = record["batch"]
    assert batch["tokens"] == batch["decode_tokens"] + batch["prefill_tokens"]
    assert batch["requests"] == batch["decode_requests"] + batch["prefill_requests"]
    if record["policy"] == "slot_only":
        return
    # This experiment uses k=7. Production shaping uses the configured width.
    assert batch["max_decode_query"] <= 8
    assert batch["gdn"]["num_prefill_tokens"] == batch["prefill_tokens"]
    assert batch["gdn"]["num_spec_decode_tokens"] == batch["decode_tokens"]
    assert batch["gdn"].get("null_sources", 0) == 0
    if record["tokens"] > 4096 or record["policy"] == "prefill":
        assert batch["prefill_requests"] > 0
        assert batch["mode"] != "FULL"


def kernel_rows(root: Path, batch: dict) -> list[dict]:
    label = f"{batch['policy']}-{batch['tokens']}"
    trace = json.loads((root / f"{label}-trace.json").read_text())
    grouped = defaultdict(list)
    for event in trace["traceEvents"]:
        if event.get("cat") != "kernel":
            continue
        dimensions = {k: v for k, v in event.get("args", {}).items() if k in ("grid", "block")}
        grouped[(event["name"], json.dumps(dimensions, sort_keys=True))].append(event["dur"])
    return [dict(policy=batch["policy"], tokens=batch["tokens"], kernel=name,
                 dimensions=dimensions, count=len(durations), total_us=sum(durations))
            for (name, dimensions), durations in grouped.items()]


if __name__ == "__main__":
    main()
