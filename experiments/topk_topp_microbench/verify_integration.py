"""Use the real rejection kernel to test seeded output compatibility."""

import argparse
import json
from pathlib import Path

import torch

from validate import candidate, native, captured_cases
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=50000)
    parser.add_argument("--backends", nargs="+", default=["pytorch", "flashinfer"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case, source, ks, ps in captured_cases(args.captures):
        rows.extend(check_capture(case, source, ks, ps, args.trials, args.backends))
        (args.output / "integration.json").write_text(json.dumps(rows, indent=2))
    (args.output / "DONE").write_text("complete\n")
    raise SystemExit(2 if any(not r["same_seed_output"] or not r["statistics_valid"] for r in rows) else 0)


def check_capture(case: dict, source: torch.Tensor, ks: list[int], ps: list[float], trials: int, backends: list[str]) -> list[dict]:
    k = torch.tensor(ks, dtype=torch.int32, device="cuda")
    p = torch.tensor(ps, device="cuda")
    original = native.apply_top_k_top_p_triton(source.clone(), k, p)
    rows = []
    for backend in backends:
        actual, route = candidate(source.clone(), ks, k, p, backend)
        different = (original != actual).any(-1).nonzero().flatten()
        selected = int(different[0]) if different.numel() else 0
        support = ((original > -float("inf")) | (actual > -float("inf")))[selected]
        # Compress only zero-probability vocabulary entries. Both paths use the same
        # token map and Philox keys. This validates verification, not full-vocab timing.
        left = original[selected, support]
        right = actual[selected, support]
        token = int(left.argmax())
        inputs = verification_inputs(left, trials, token)
        expected, expected_count = rejection_sample(**inputs)
        inputs["target_logits"] = right.expand(trials * 4, -1).contiguous()
        observed, observed_count = rejection_sample(**inputs)
        valid = torch.arange(4, device="cuda").unsqueeze(0) < expected_count.unsqueeze(-1)
        same = torch.equal(expected_count, observed_count) and torch.equal(expected[valid], observed[valid])
        z_native = multinomial_z(expected[:, 0], left.softmax(-1))
        z_candidate = multinomial_z(observed[:, 0], right.softmax(-1))
        row = dict(capture=case["capture"], original_rows=case["original_rows"], selected_row=selected, backend=backend, route=route,
                   trials=trials, union_support=int(support.sum()), same_seed_output=bool(same),
                   first_token_difference=float((expected[:, 0] != observed[:, 0]).float().mean()),
                   count_difference=float((expected_count != observed_count).float().mean()),
                   native_max_z=z_native, candidate_max_z=z_candidate,
                   statistics_valid=z_native < 7 and z_candidate < 7)
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def verification_inputs(logits: torch.Tensor, trials: int, token: int) -> dict:
    device = logits.device
    local = torch.arange(4, device=device, dtype=torch.int32)
    drafts = torch.full((trials, 4), token, device=device, dtype=torch.int64)
    drafts[:, 0] = 0
    requests = torch.arange(trials, device=device, dtype=torch.int32)
    return dict(target_logits=logits.expand(trials * 4, -1).contiguous(), draft_logits_cache=None,
                draft_sampled=drafts.flatten(), cu_num_logits=torch.arange(trials + 1, device=device, dtype=torch.int32) * 4,
                pos=local.repeat(trials), idx_mapping=requests,
                expanded_idx_mapping=requests.repeat_interleave(4), expanded_local_pos=local.repeat(trials),
                temperature=torch.full((trials,), 0.7, device=device),
                seed=torch.arange(trials, device=device, dtype=torch.int64) + 42, num_speculative_steps=3)


def multinomial_z(samples: torch.Tensor, probs: torch.Tensor) -> float:
    observed = torch.bincount(samples, minlength=probs.numel()).double()
    expected = probs.double() * samples.numel()
    scale = (samples.numel() * probs.double() * (1 - probs.double())).sqrt()
    z = (observed - expected).abs() / scale.clamp_min(1e-12)
    return float(z.max())


if __name__ == "__main__":
    main()
