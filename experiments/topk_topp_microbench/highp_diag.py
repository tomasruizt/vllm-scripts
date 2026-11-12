"""Per-row diagnosis of top-p masks at high p: kept count vs. the FP64 exact count, and
how far the kept mass is from p (mass before the last kept logit, mass reached).

Usage: highp_diag.py ROWS.pt P BATCH MODULE... (MODULE_kernel.py, or "main" for triton_kernel.py)
"""

import importlib
import sys

import torch


def main() -> None:
    path, p, batch, modules = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
    rows = torch.load(path).float().cuda()
    logits = rows.repeat((batch + len(rows) - 1) // len(rows), 1)[:batch].contiguous()
    ps = torch.full((batch,), p, device="cuda")
    exact = exact_counts(logits, p)
    for name in modules:
        module = importlib.import_module("triton_kernel" if name == "main" else f"{name}_kernel")
        kept = module.apply_top_k_top_p_triton(logits.clone(), None, ps) > -float("inf")
        report(name, logits, kept, exact, p)


def exact_counts(logits: torch.Tensor, p: float) -> torch.Tensor:
    probs, _ = torch.sort(torch.softmax(logits.double(), -1), dim=-1, descending=True, stable=True)
    before = probs.cumsum(-1) - probs
    return (before < p).sum(-1)


def report(name: str, logits: torch.Tensor, kept: torch.Tensor, exact: torch.Tensor, p: float) -> None:
    probs, order = torch.sort(torch.softmax(logits.double(), -1), dim=-1, descending=True, stable=True)
    cum = probs.cumsum(-1)
    count = kept.sum(-1)
    prefix = torch.arange(logits.shape[-1], device="cuda") < count[:, None]
    not_prefix = (torch.gather(kept, 1, order) != prefix).any(-1)
    last = (count - 1).clamp(min=0)[:, None]
    reach = cum.gather(1, last).squeeze(1) - p
    before = reach - probs.gather(1, last).squeeze(1)
    bad = (count != exact) | not_prefix
    print(f"{name}: {int(bad.sum())} of {len(count)} rows differ from the exact count")
    for r in bad.nonzero().flatten().tolist()[:6]:
        print(f"  row {r}: kept {int(count[r])} exact {int(exact[r])} prefix {not bool(not_prefix[r])}"
              f" reach-p {reach[r].item():+.2e} before-p {before[r].item():+.2e}")


if __name__ == "__main__":
    main()
