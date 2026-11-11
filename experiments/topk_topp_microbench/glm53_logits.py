"""Stack the first 14 verify-only captures (rows a multiple of MTP5's 6) of capture_probe.py
into a [112, vocab] FP32 tensor like qwen36-verify-logits-t1.pt, and print its stats.
Usage: glm53_logits.py CAPTURE_DIR OUTPUT"""
import json
import sys
from pathlib import Path

import torch


def main() -> None:
    captures = [torch.load(f) for f in sorted(Path(sys.argv[1]).glob("run*/logits-*.pt"))]
    verify = [c for c in captures if c["original_rows"] % 6 == 0][:14]
    assert len(verify) == 14, [c["original_rows"] for c in captures]
    logits = torch.cat([c["logits"] for c in verify]).float().contiguous()
    torch.save(logits, sys.argv[2])
    p = torch.cat([c["p"] for c in verify]) if verify[0]["p"] is not None else None
    stats = dict(rows=logits.shape[0], vocab=logits.shape[1], captures=len(captures),
                 original_rows=[c["original_rows"] for c in verify],
                 dtypes=sorted({c["original_dtype"] for c in verify}),
                 k=sorted({None if c["k"] is None else tuple(c["k"].unique().tolist()) for c in verify}, key=str),
                 p=None if p is None else p.unique().tolist(), nonfinite=int((~logits.isfinite()).sum()),
                 **kept_stats(logits, 0.95))
    print(json.dumps(stats))


def kept_stats(logits: torch.Tensor, p: float) -> dict:
    """Tokens in the smallest prefix of mass >= p (exact FP64, stable order)."""
    probs = logits.double().softmax(-1).sort(dim=-1, descending=True, stable=True).values
    kept = ((probs.cumsum(-1) - probs) < p).sum(-1)
    return dict(kept_median=int(kept.median()), kept_max=int(kept.max()), kept_min=int(kept.min()),
                kept_mean=float(kept.double().mean()))


if __name__ == "__main__":
    main()
