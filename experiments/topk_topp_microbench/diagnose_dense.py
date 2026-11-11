"""Inspect disabled top-p rows through the public FlashInfer renormalization API."""

import argparse
import json
from pathlib import Path
import runpy

import flashinfer
import flashinfer.sampling as sampling
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    module = runpy.run_path(str(args.benchmark), run_name="benchmark_helpers")
    rows = []
    for vocab in [32768, 131072, 151936, 248320, 262144]:
        torch.manual_seed(0)
        logits = module["create_logits"](4, vocab)
        probs = logits.softmax(-1)
        p = torch.tensor([0.8, 0.8, 1.0, 1.0], device="cuda")
        for repeat in range(2):
            filtered = sampling.top_p_renorm_probs(probs.clone(), p, is_deterministic=True)
            kept = filtered > 0
            mass = torch.where(kept, probs, 0).sum(-1)
            for row in range(4):
                item = dict(vocab=vocab, row=row, p=float(p[row]), repeat=repeat,
                            kept=int(kept[row].sum()), input_sum=float(probs[row].sum()),
                            retained_mass=float(mass[row]), output_sum=float(filtered[row].sum()),
                            flashinfer=flashinfer.__version__)
                rows.append(item)
                print(json.dumps(item), flush=True)
    (args.output / "diagnostic.json").write_text(json.dumps(rows, indent=2))
    (args.output / "DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
