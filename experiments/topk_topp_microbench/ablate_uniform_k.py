"""Isolate host-k hints using the existing vLLM benchmark and identical inputs."""

import argparse
from contextlib import redirect_stdout
import importlib.util
import json
from pathlib import Path
import sys
import time

from sweep_existing import metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("topk_benchmark", "benchmark_topk_topp.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original_prepare = module.prepare_kwargs
    module.prepare_kwargs = lambda name, k, p, vocab: prepare(
        original_prepare, name, k, p, vocab
    )
    for name in ["fi-uniform", "fi-bound", "fi-cap256"]:
        module.IMPLEMENTATIONS[name] = module.apply_top_k_top_p_flashinfer_fastpath
    (args.output / "metadata.json").write_text(json.dumps(metadata(), indent=2))
    started = time.monotonic()
    for repeat in range(3):
        execute(module, args.output, repeat)
    (args.output / "summary.json").write_text(json.dumps({"elapsed_s": time.monotonic() - started}))
    (args.output / "DONE").write_text("ok\n")


def prepare(original, name: str, k, p, vocab: int) -> dict:
    if not name.startswith("fi-"):
        return original(name, k, p, vocab)
    hints = original("flashinfer-fastpath", k, p, vocab)
    value = hints["top_k_max"]
    if name == "fi-uniform":
        return {"top_k_min": value, "top_k_max": value}
    # A conservative lower bound forces masking even on uniform input.
    return {"top_k_min": 1, "top_k_max": value if name == "fi-bound" else 256}


def execute(module, output: Path, repeat: int) -> None:
    # Rotate all four implementations across cells, reversing the second pass.
    names = ["triton", "fi-uniform", "fi-bound", "fi-cap256"]
    names = names[repeat:] + names[:repeat]
    if repeat == 1:
        names.reverse()
    command = ["benchmark_topk_topp.py", "--batch-sizes", "4", "64", "512",
               "--vocab-sizes", "131072", "248320", "--scenarios", "topk_whole",
               "topk_topp_whole", "--implementations", *names, "--check",
               "--warmup-iters", "5", "--benchmark-iters", "30", "--output",
               str(output / f"case-{repeat:03d}.jsonl")]
    (output / f"case-{repeat:03d}.command.json").write_text(json.dumps(command, indent=2))
    print(json.dumps({"repeat": repeat, "order": names}), flush=True)
    sys.argv = command
    with (output / f"case-{repeat:03d}.log").open("x") as log, redirect_stdout(log):
        module.main()


if __name__ == "__main__":
    main()
