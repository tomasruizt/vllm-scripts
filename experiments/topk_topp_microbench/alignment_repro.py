"""Isolate the unaligned-vocabulary FlashInfer error in its own CUDA process."""

import argparse
import json
from pathlib import Path

import flashinfer
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph-safe", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    logits = torch.randn(4, 257, device="cuda") / 0.7
    print(json.dumps(dict(shape=list(logits.shape), stride=list(logits.stride()),
                          dtype=str(logits.dtype), graph_safe=args.graph_safe,
                          flashinfer=flashinfer.__version__)), flush=True)
    values, indices = flashinfer.top_k(logits, 1, sorted=True, dsa_graph_safe=args.graph_safe)
    torch.cuda.synchronize()
    expected = logits.max(-1).values
    assert torch.equal(values[:, 0], expected)
    (args.output / "DONE").write_text("ok\n")
    print("correct top1 values", flush=True)


if __name__ == "__main__":
    main()
