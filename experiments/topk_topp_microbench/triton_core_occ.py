"""Register counts and computed capacity of New Triton's persistent kernel per config,
and whether an oversubscribed cooperative launch is rejected (no kernel is run
otherwise). Usage, from the PR tree: python triton_core_occ.py
"""

import torch
import triton_core


def main() -> None:
    for tiers in (2, 1):
        triton_core.P_TIERS = tiers
        for rows in (16, 64, 256, 2048):
            logits = torch.randn(rows, 248320, device="cuda")
            p = torch.full((rows,), 0.95, device="cuda")
            triton_core._CAPACITY.clear()
            triton_core.DRY_RUN = True
            triton_core.apply_top_k_top_p_triton_core(logits, None, p)
            triton_core.DRY_RUN = False
            seen = {"grid": triton_core.LAUNCHED["programs"]}
            (key, cap), = triton_core._CAPACITY.items()
            kernel = triton_core.LAST_WARMUP
            print(f"tiers={tiers} rows={rows} regs={kernel.n_regs} smem={kernel.metadata.shared} capacity={cap} grid={seen['grid']}", flush=True)


if __name__ == "__main__":
    main()
