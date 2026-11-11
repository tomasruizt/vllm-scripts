"""Retain small counterexamples for captured-logit disagreements."""

import argparse
import json
from pathlib import Path

import torch

from validate import captured_cases, candidate, native, apply_top_k_top_p_pytorch, synthetic_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--captures", type=Path)
    parser.add_argument("--phase", choices=["captured", "large"], default="captured")
    parser.add_argument("--backends", nargs="+", default=["pytorch", "flashinfer"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = captured_cases(args.captures) if args.phase == "captured" else synthetic_cases("large")
    for case, source, ks, ps in cases:
        if args.phase == "large" and not (case["seed"] == 0 and case["kind"] == "normal"
                                          and case["batch"] == 256 and case["top_k"] in (20, 64) and case["top_p"] == 0.8):
            continue
        k = torch.tensor(ks, dtype=torch.int32, device="cuda")
        p = torch.tensor(ps, device="cuda")
        original = native.apply_top_k_top_p_triton(source.clone(), k, p)
        reference = apply_top_k_top_p_pytorch(source.clone(), k, p)
        for backend in args.backends:
            actual, route = candidate(source.clone(), ks, k, p, backend)
            for row in (actual != original).any(-1).nonzero().flatten().tolist():
                values, indices = source[row].topk(max(32, ks[row] + 4), sorted=True)
                probabilities = values[:ks[row]].double().softmax(-1)
                native_only = torch.isfinite(original[row]) & ~torch.isfinite(actual[row])
                candidate_only = torch.isfinite(actual[row]) & ~torch.isfinite(original[row])
                item = dict(case=case, capture=case.get("capture"), backend=backend, row=row, p=float(p[row]),
                            ids=indices.cpu().tolist(), logits=values.cpu().tolist(),
                            native_kept=torch.isfinite(original[row, indices]).cpu().tolist(),
                            reference_kept=torch.isfinite(reference[row, indices]).cpu().tolist(),
                            candidate_kept=torch.isfinite(actual[row, indices]).cpu().tolist(),
                            native_only_logits=source[row, native_only].cpu().tolist(),
                            candidate_only_logits=source[row, candidate_only].cpu().tolist(),
                            topk_probabilities_f64=probabilities.cpu().tolist(),
                            topk_prefix_f64=probabilities.cumsum(-1).cpu().tolist())
                rows.append(item)
                print(json.dumps(dict(capture=item["capture"], backend=backend, row=row,
                                      native=sum(item["native_kept"]), reference=sum(item["reference_kept"]),
                                      candidate=sum(item["candidate_kept"]))), flush=True)
    (args.output / "explanations.json").write_text(json.dumps(rows, indent=2))
    (args.output / "DONE").write_text("complete\n")


if __name__ == "__main__":
    main()
