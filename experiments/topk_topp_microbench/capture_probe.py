"""Qwen logit capture and a controlled filtering A/B worker extension."""

from pathlib import Path
from functools import partial

import torch


class SolProbe:
    def init_worker_extension(self) -> list[str]:
        if hasattr(self, "capture_state"):
            return ["capture already installed"]
        from vllm.v1.sample.ops import topk_topp_sampler
        self.capture_state = dict(path=None, calls=0, saved=0, minimum=2, mode="native")
        original = topk_topp_sampler.apply_top_k_top_p_triton
        import flashinfer
        from benchmark import compact_filter
        select = partial(flashinfer.top_k, deterministic=True, tie_break=1)

        def capture(logits, k, p, *args, **kwargs):
            self.capture_logits(logits, k, p)
            if self.capture_state["mode"] == "flashinfer-deterministic" and k is not None:
                return compact_filter(logits, select, p, 20)
            return original(logits, k, p, *args, **kwargs)

        topk_topp_sampler.apply_top_k_top_p_triton = capture
        return ["pre-filter capture installed"]

    def arm_capture(self, path: str, minimum: int) -> dict:
        self.capture_state.update(path=path, calls=0, saved=0, minimum=minimum)
        Path(path).mkdir(parents=True, exist_ok=True)
        return self.capture_state

    def capture_status(self) -> dict:
        return self.capture_state

    def set_filter_mode(self, mode: str) -> str:
        assert mode in ("native", "flashinfer-deterministic")
        self.capture_state.update(mode=mode, path=None)
        return mode

    def capture_logits(self, logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> None:
        state = self.capture_state
        if not state["path"] or state["saved"] >= 8 or logits.shape[0] < state["minimum"]:
            return
        state["calls"] += 1
        if state["calls"] % 16:
            return
        rows = logits.shape[0]
        indices = torch.linspace(0, rows - 1, min(rows, 8), device=logits.device).long()
        item = dict(logits=logits[indices].detach().cpu(), original_rows=rows,
                    indices=indices.cpu(), original_dtype=str(logits.dtype),
                    k=k[indices].cpu() if k is not None else None,
                    p=p[indices].cpu() if p is not None else None)
        torch.save(item, Path(state["path"]) / f"logits-{state['saved']:03d}.pt")
        state["saved"] += 1
