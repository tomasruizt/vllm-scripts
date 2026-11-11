"""Fast diagnostics for the v2 unified filter on small cases."""

import argparse
import json
import struct
from pathlib import Path

import torch

from unified_topk_topp_v2 import UnifiedFilter

DBG = 16 + 128
FIELDS = ["prefix", "above", "cands", "cut", "keep", "target", "bucket", "mask"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7)
    prototype = UnifiedFilter()
    vocab = 4096
    logits = torch.randn(4, vocab, device="cuda")
    for name, k, p in cases():
        debug = torch.zeros(DBG * logits.shape[0], device="cuda", dtype=torch.int64)
        out = prototype(logits.clone(), k, p, debug=debug)
        report(name, logits, out)
        state = debug.view(-1, DBG)[0]
        print(json.dumps(dict(case=name, state=dict(zip(FIELDS, state[:8].tolist())),
                              hits=int(state[8]), k=int(state[9]))), flush=True)
        dump_candidates(state)
        print(json.dumps(dict(levels=state[32:48].tolist())), flush=True)


def cases():
    p = torch.tensor([0.8] * 4, device="cuda")
    k20 = torch.tensor([20] * 4, device="cuda", dtype=torch.int32)
    return [("topk20", k20, None), ("both", k20, p)]


def dump_candidates(state) -> None:
    keys = state[16:].tolist()
    values = [decode(key) for key in keys[:24] if key]
    print(json.dumps(dict(candidate_values=values)), flush=True)


def decode(packed: int) -> float:
    key = packed >> 32
    bits = ((key ^ 0x80000000) if key & 0x80000000 else (~key & 0xFFFFFFFF)) & 0xFFFFFFFF
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def report(name, source, actual) -> None:
    order = source.argsort(dim=-1, descending=True, stable=True)
    kept = actual.isfinite().gather(-1, order)
    counts = kept.sum(-1).tolist()
    ranks = torch.arange(source.shape[1], device=source.device)
    counted = torch.tensor(counts, device=source.device)
    contiguous = bool(torch.equal(kept, ranks[None, :] < counted[:, None]))
    top = source.sort(dim=-1, descending=True).values[0][:22].tolist()
    print(json.dumps(dict(case=name, counts=counts, contiguous=contiguous,
                          top_values=top)), flush=True)


if __name__ == "__main__":
    main()
