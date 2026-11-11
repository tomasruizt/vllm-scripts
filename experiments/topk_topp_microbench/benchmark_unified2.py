"""Validate and time the v3 unified filter against the production Triton kernel."""

import argparse
from contextlib import redirect_stdout
import importlib.util
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path
from threading import Timer

from benchmark_unified import validate
from sweep_existing import metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--batches", default="4,64,512")
    parser.add_argument("--vocabs", default="65536,131072,248320")
    parser.add_argument("--skip-topp-only", action="store_true")
    parser.add_argument("--kernel-module", default="unified_topk_topp_v3")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    warnings.filterwarnings("error", message=".*Falling back to CUDA events.*")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    prototype = __import__(args.kernel_module).UnifiedFilter()
    print(json.dumps(dict(compiled_s=round(time.monotonic() - started, 1))), flush=True)
    module = load_benchmark()
    info = metadata()
    for filename in [f"{args.kernel_module}.cu", f"{args.kernel_module}.py", "benchmark_unified2.py"]:
        info["sources"][filename] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    (args.output / "metadata.json").write_text(json.dumps(info, indent=2))
    if args.skip_topp_only:
        original = module.create_benchmark_configs
        module.create_benchmark_configs = lambda *a, **kw: [
            config for config in original(*a, **kw)
            if "topk" in config.name or "mixed" in config.name]
    module.IMPLEMENTATIONS["unified"] = prototype
    watchdog = Timer(120, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    try:
        rows = validate(module, prototype, args.smoke_only)
    finally:
        watchdog.cancel()
    (args.output / "validation.json").write_text(json.dumps(rows, indent=2))
    failures = [row for row in rows if not row["valid"]]
    print(json.dumps(dict(cases=len(rows), failures=len(failures))), flush=True)
    if failures:
        print(json.dumps(dict(failed_cases=failures[:5])), flush=True)
        fallbacks = sum(not row.get("reference_close", True) for row in rows)
        print(json.dumps(dict(reference_mismatch=fallbacks)), flush=True)
        raise SystemExit(2)
    if args.timing:
        started = time.monotonic()
        run_timing(module, args)
        (args.output / "summary.json").write_text(json.dumps(dict(elapsed_s=time.monotonic() - started)))
    (args.output / "DONE").write_text("ok\n")


def load_benchmark():
    spec = importlib.util.spec_from_file_location("benchmark_helpers", "benchmark_topk_topp.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_timing(module, args) -> None:
    batches = [int(x) for x in args.batches.split(",")]
    vocabs = [int(x) for x in args.vocabs.split(",")]
    for repeat in range(args.repeats):
        order = ["unified", "triton"] if repeat == 1 else ["triton", "unified"]
        command = ["benchmark_topk_topp.py", "--batch-sizes", *map(str, batches),
                   "--vocab-sizes", *map(str, vocabs), "--top-k", "20", "--top-p", "0.8",
                   "--implementations", *order, "--check", "--warmup-iters", "5",
                   "--benchmark-iters", "20", "--output",
                   str(args.output / f"case-{repeat:03d}.jsonl")]
        (args.output / f"case-{repeat:03d}.command.json").write_text(json.dumps(command, indent=2))
        sys.argv = command
        print(json.dumps(dict(repeat=repeat, order=order)), flush=True)
        with (args.output / f"case-{repeat:03d}.log").open("x") as log, redirect_stdout(log):
            module.main()


if __name__ == "__main__":
    main()
