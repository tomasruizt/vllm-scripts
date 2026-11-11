"""Validate benchmark dispatch hints and confirm their effect in a GPU trace."""

import argparse
import json
from pathlib import Path
import runpy

import torch

from benchmark import profile_kernels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    module = runpy.run_path(str(args.benchmark), run_name="benchmark_helpers")
    rows = []
    for config in module["create_benchmark_configs"]([4, 64], [32768]):
        rows.append(check_case(module, config, args.output))
    config = module["create_benchmark_configs"]([4], [32768])[4]
    rows.append(check_case(module, config, args.output, grammar=True))
    (args.output / "validation.json").write_text(json.dumps(rows, indent=2))
    (args.output / "DONE").write_text("ok\n")
    print(json.dumps(dict(cases=len(rows), all_exact=all(r["legacy_exact"] for r in rows))), flush=True)


def check_case(module: dict, config, output: Path, grammar: bool = False) -> dict:
    torch.manual_seed(0)
    logits = module["create_logits"](config.batch_size, config.vocab_size)
    if grammar:
        logits[1, 50:] = -float("inf")
    k = config.k_values
    p = config.p_values
    scenario = config.name.split("_b")[0]
    expected_hint = scenario in {"topp_whole", "topp_partial", "mixed_partial"}
    kwargs = module["prepare_kwargs"]("triton", k, p, config.vocab_size)
    assert kwargs == {"has_topp_only_rows": expected_hint}, (scenario, kwargs)
    func = module["IMPLEMENTATIONS"]["triton"]
    legacy = lambda: func(logits.clone(), k, p)
    corrected = lambda: func(logits.clone(), k, p, **kwargs)
    exact = torch.equal(legacy(), corrected())
    assert exact, (config.name, grammar)
    row = dict(scenario=scenario, batch=config.batch_size, grammar=grammar,
               has_topp_only_rows=expected_hint, legacy_exact=exact)
    if config.batch_size == 4 and scenario in {"topk_topp_whole", "mixed_partial"} and not grammar:
        row["legacy_split_launches"] = split_launches(legacy, output / f"{scenario}-legacy.json")
        row["corrected_split_launches"] = split_launches(corrected, output / f"{scenario}-corrected.json")
        assert row["legacy_split_launches"] == 7
        assert row["corrected_split_launches"] == (7 if expected_hint else 0)
    print(json.dumps(row), flush=True)
    return row


def split_launches(call, output: Path) -> float:
    kernels = profile_kernels(call, output)
    return sum(len(times) for name, times in kernels.items() if "_topp_sb_" in name) / 20


if __name__ == "__main__":
    main()
