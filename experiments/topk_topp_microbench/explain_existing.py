"""Replay the failing Gaussian benchmark fixtures without the timing loop."""

import argparse
import json
from pathlib import Path
import runpy

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    module = runpy.run_path(str(args.benchmark), run_name="benchmark_helpers")
    plans = json.loads(args.plan.read_text())
    rows = []
    for index in (148, 149, 150):
        item = plans[index]
        torch.manual_seed(0)
        configs = module["create_benchmark_configs"](item["batches"], item["vocabs"], top_k=item["top_k"], top_p=item["top_p"])
        for config in configs:
            if config.name.split("_b")[0] not in item["scenarios"]:
                continue
            source = module["create_logits"](config.batch_size, config.vocab_size, distribution=item["distribution"])
            k = item["top_k"]
            kwargs = module["prepare_kwargs"]("flashinfer-fastpath", config.k_values, config.p_values, config.vocab_size)
            actual = module["apply_top_k_top_p_flashinfer_fastpath"](source.clone(), config.k_values, config.p_values, **kwargs)
            reference = module["apply_top_k_top_p_pytorch"](source.clone(), config.k_values, config.p_values)
            for row in (actual != reference).any(-1).nonzero().flatten().tolist():
                a = torch.isfinite(actual[row])
                r = torch.isfinite(reference[row])
                vals = source[row].topk(k + 1).values.double()
                cdf = vals[:k].softmax(-1).cumsum(-1)
                n = int(a.sum())
                left = source[row, a & ~r].sort().values
                right = source[row, r & ~a].sort().values
                result = dict(case_index=index, scenario=config.name, row=row, top_k=k,
                              top_p=float(config.p_values[row]) if config.p_values is not None else None,
                              actual_kept=n, reference_kept=int(r.sum()),
                              kth_tied=bool(vals[k-1] == vals[k]),
                              equal_logit_swap=bool(torch.equal(left, right)),
                              prefix_mass=float(cdf[min(n, k)-1]),
                              previous_mass=float(cdf[n-2]) if n>1 else 0.0,
                              actual_only_logits=left.cpu().tolist(), reference_only_logits=right.cpu().tolist())
                rows.append(result)
                print(json.dumps(result), flush=True)
    (args.output / "explanations.json").write_text(json.dumps(rows, indent=2))
    (args.output / "DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
