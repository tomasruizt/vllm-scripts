"""FlashInfer renorm-kernel pipelines, adapted to the sampler's masked-logit contract."""

import flashinfer.sampling as fi_sampling
import torch


def apply_fi_mask(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> torch.Tensor:
    """Benchmark helper on main: top_k_mask_logits, softmax, top_p_renorm_probs, mask."""
    if k is not None:
        logits = fi_sampling.top_k_mask_logits(logits, k)
    if p is None:
        return logits
    probs = fi_sampling.top_p_renorm_probs(logits.softmax(dim=-1), p, is_deterministic=True)
    return logits.masked_fill(probs == 0, -float("inf"))


def apply_fi_renorm(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> torch.Tensor:
    """Proposed pipeline with reproducible accumulation, returning masked logits."""
    probs = renorm_probs(logits, k, p, is_deterministic=True)
    return logits.masked_fill(probs == 0, -float("inf"))


def apply_fi_renorm_nondet(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> torch.Tensor:
    """Proposed pipeline with the API's default float accumulation."""
    probs = renorm_probs(logits, k, p, is_deterministic=False)
    return logits.masked_fill(probs == 0, -float("inf"))


def apply_fi_renorm_probs(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> torch.Tensor:
    """Best-case timing: API defaults and no masked-logit reconstruction. Not a drop-in."""
    return renorm_probs(logits, k, p, is_deterministic=False)


def renorm_probs(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None, is_deterministic: bool
) -> torch.Tensor:
    """softmax, then the separate top-k and top-p renorm kernels."""
    probs = logits.softmax(dim=-1)
    if k is not None:
        probs = fi_sampling.top_k_renorm_probs(probs, k, is_deterministic=is_deterministic)
    if p is not None:
        probs = fi_sampling.top_p_renorm_probs(probs, p, is_deterministic=is_deterministic)
    return probs


RENORM_IMPLEMENTATIONS = {
    "fi-mask": apply_fi_mask,
    "fi-renorm": apply_fi_renorm,
    "fi-renorm-nondet": apply_fi_renorm_nondet,
    "fi-renorm-probs": apply_fi_renorm_probs,
}
