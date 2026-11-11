"""Compare optimized masked logits with the preserved adapter on edge inputs."""

import argparse
import itertools
import json
from pathlib import Path
import runpy

import torch

from benchmark import edge_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = runpy.run_path("baseline.py")["apply_top_k_top_p_flashinfer_fastpath"]
    optimized = runpy.run_path("benchmark_topk_topp.py")["apply_top_k_top_p_flashinfer_fastpath"]
    torch.manual_seed(13)
    rows = []
    for (label, source, cutoff), heterogeneous, strided in itertools.product(
        edge_cases(65536), (False, True), (False, True)
    ):
        logits = torch.full((4, 65536), -float("inf"), device="cuda")
        logits[:, :source.shape[1]] = source
        if strided:
            logits = logits.repeat_interleave(2, -1)[:, ::2]
        ks = [1, 3, 10, 20] if heterogeneous else [20] * 4
        k = torch.tensor(ks, device="cuda", dtype=torch.int32)
        for p_values in (None, [cutoff] * 4, [0.1, 0.5, 0.8, 1.0]):
            p = None if p_values is None else torch.tensor(p_values, device="cuda")
            if strided:
                k = torch.repeat_interleave(k, 2)[::2]
                p = None if p is None else torch.repeat_interleave(p, 2)[::2]
            kwargs = dict(top_k_min=min(ks), top_k_max=max(ks))
            expected = baseline(logits, k, p, **kwargs)
            actual = optimized(logits, k, p, **kwargs)
            valid = torch.allclose(actual, expected, rtol=0, atol=0, equal_nan=True)
            row = dict(case=label, heterogeneous=heterogeneous, strided=strided,
                       p=p_values, valid=bool(valid))
            rows.append(row)
            print(json.dumps(row), flush=True)
    (args.output / "edges.json").write_text(json.dumps(rows, indent=2))
    failures = sum(not r["valid"] for r in rows)
    print(json.dumps(dict(cases=len(rows), failures=failures)), flush=True)
    raise SystemExit(2 if failures else 0)


if __name__ == "__main__":
    main()
