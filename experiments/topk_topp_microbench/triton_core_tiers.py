"""Which tier of the Triton core prototype would take each real top-p-only row
(p = 0.95), replicated in PyTorch: nucleus size, and the wide tier's threshold choice.

Usage: python triton_core_tiers.py --logits-file ROWS.pt
"""

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-file", required=True)
    parser.add_argument("--p", type=float, default=0.95)
    args = parser.parse_args()
    rows = torch.load(args.logits_file).float().cuda()
    stats = [row_stats(row, args.p) for row in rows]
    keys = stats[0].keys()
    table = {key: torch.tensor([s[key] for s in stats], dtype=torch.float64) for key in keys}
    n = table["nucleus"]
    print(f"rows {len(rows)}, nucleus median {n.median():.0f} p90 {n.quantile(0.9):.0f} p99 {n.quantile(0.99):.0f} max {n.max():.0f}")
    print(f"nucleus <= 32: {(n <= 32).double().mean():.3f}, <= 128: {(n <= 128).double().mean():.3f}")
    wide = n > 128
    for key in ("pick", "groups", "candidates", "sure", "ok"):
        values = table[key][wide]
        print(key, "over wide rows: mean", f"{values.mean():.2f}", "max", f"{values.max():.0f}")
    print("wide rows", int(wide.sum()), "wide ok", int(table["ok"][wide].sum()))


def row_stats(row: torch.Tensor, p: float) -> dict:
    vocab = row.numel()
    order = row.sort(descending=True).values
    weight = (order - order[0]).exp()
    target = p * weight.sum()
    nucleus = int((weight.cumsum(0) - weight < target).sum())
    padded = torch.full((-(-vocab // 256) * 256,), -float("inf"), device=row.device)
    padded[:vocab] = row
    level1 = padded.view(-1, 16).amax(1)
    level2 = padded.view(-1, 256).amax(1)
    supergroups = level2.numel()
    ranks = [max((j + 1) * supergroups // 8, 1) - 1 for j in range(8)]
    thresholds = level2.sort(descending=True).values[ranks]
    counts = (level1[None, :] >= thresholds[:, None]).sum(1)
    masses = torch.where(level1[None, :] >= thresholds[:, None], (level1 - order[0]).exp()[None, :], 0).sum(1)
    sure = [j for j in range(8) if masses[j] >= target]
    fits = [j for j in range(8) if counts[j] <= 1024]
    pick = sure[0] if sure else (fits[-1] if fits else -1)
    t = thresholds[pick] if pick >= 0 else -float("inf")
    groups = int((level1 >= t).sum())
    candidates = int((row >= t).sum())
    mass = (row[row >= t] - order[0]).exp().sum()
    ok = pick >= 0 and groups <= 1024 and candidates <= 4096 and mass >= target
    return dict(nucleus=nucleus, pick=pick, groups=groups, candidates=candidates, sure=bool(sure), ok=ok)


if __name__ == "__main__":
    main()
