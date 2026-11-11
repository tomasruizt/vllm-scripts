"""Run upstream sampler tests unchanged, with an optional experimental adapter."""

import argparse
from collections import Counter
from functools import partial
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["native", "flashinfer-deterministic"])
    parser.add_argument("--tests", nargs="+", default=["tests/v1/sample/test_topk_topp_sampler.py",
                                                     "tests/v1/spec_decode/test_rejection_sampler_utils.py"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.backend:
        raise SystemExit(run_tests(args.output, args.backend, args.tests))
    codes = {}
    for backend in ("native", "flashinfer-deterministic"):
        command = [sys.executable, __file__, "--output", str(args.output), "--backend", backend]
        command += ["--tests", *args.tests]
        with (args.output / f"{backend}.log").open("x") as log:
            codes[backend] = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    (args.output / "exit-codes.json").write_text(json.dumps(codes, indent=2))
    (args.output / "DONE").write_text("complete\n")
    raise SystemExit(max(codes.values()))


def run_tests(output: Path, backend: str, selection: list[str]) -> int:
    import flashinfer
    import pytest
    from benchmark import compact_filter, native
    from vllm.v1.sample.ops import topk_topp_sampler

    original = native.apply_top_k_top_p_triton
    routes = Counter()
    select = partial(flashinfer.top_k, deterministic=True, tie_break=1)

    def adapter(logits, k, p, *args, **kwargs):
        # CPU readback is test-only eligibility, excluded from performance runs.
        ks = k.tolist() if k is not None else []
        eligible = ks and 0 < ks[0] <= 128 and ks[0] < logits.shape[-1] and len(set(ks)) == 1
        aligned = logits.shape[0] == 1 or logits.shape[-1] % 4 == 0
        if eligible and aligned:
            routes["compact"] += 1
            return compact_filter(logits, select, p, ks[0])
        routes["native-fallback"] += 1
        return original(logits, k, p, *args, **kwargs)

    if backend != "native":
        native.apply_top_k_top_p_triton = adapter
        topk_topp_sampler.apply_top_k_top_p_triton = adapter
    root = Path(native.__file__).resolve().parents[4]
    tests = [root / test for test in selection]
    code = pytest.main(["-q", "--tb=short", *map(str, tests)])
    (output / f"{backend}-routes.json").write_text(json.dumps(dict(routes), indent=2))
    return int(code)


if __name__ == "__main__":
    main()
