"""Check the benchmark's compact filter against an independent FP64 prefix oracle."""

import argparse
import itertools
import json
from pathlib import Path
import runpy

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--interface-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    module = runpy.run_path(str(args.benchmark), run_name="benchmark_helpers")
    interface = check_tensor_interface(module)
    (args.output / "interface.json").write_text(json.dumps(interface, indent=2))
    if args.interface_only:
        (args.output / "DONE").write_text("ok\n")
        print(json.dumps(dict(interface_cases=len(interface), failures=0)), flush=True)
        return
    rows = []
    for distribution, batch, vocab, k, p in itertools.product(
        ["normal", "bf16-rounded"], [1, 4, 128], [65536, 248320], [1, 20, 128, 256], [None, 0.1, 0.8, 1.0]
    ):
        rows.append(check(module, distribution, batch, vocab, k, p))
    (args.output / "validation.json").write_text(json.dumps(rows, indent=2))
    (args.output / "DONE").write_text("ok\n")
    print(json.dumps(dict(cases=len(rows), interface_cases=len(interface), failures=sum(not r["valid"] for r in rows),
                          reference_differences=sum(not r["reference_exact"] for r in rows))), flush=True)
    raise SystemExit(2 if any(not r["valid"] for r in rows) else 0)


def check_tensor_interface(module: dict) -> list[dict]:
    rows = []
    func = module["apply_top_k_top_p_flashinfer_fastpath"]
    for config in module["create_benchmark_configs"]([4], [65536]):
        logits = module["create_logits"](config.batch_size, config.vocab_size)
        k = config.k_values
        p = config.p_values
        assert k is None or k.device == logits.device
        assert p is None or p.device == logits.device
        kwargs = module["prepare_kwargs"]("flashinfer-fastpath", k, p, config.vocab_size)
        supported = config.name.split("_b")[0] in {"topk_whole", "topk_topp_whole"}
        if supported:
            actual = func(logits.clone(), k, p, **kwargs).softmax(-1)
            reference = module["apply_top_k_top_p_pytorch"](logits.clone(), k, p).softmax(-1)
            assert torch.allclose(actual, reference, rtol=1e-5, atol=1e-6), config.name
        else:
            expect_not_implemented(lambda: func(logits.clone(), k, p, **kwargs))
        try:
            func(logits.clone(), k, p)
        except TypeError:
            pass
        else:
            raise AssertionError("Expected missing top_k_max to raise TypeError")
        rows.append(dict(scenario=config.name, device=str(logits.device), supported=supported, valid=True))
    for hint in (0, -1, 257, 65536):
        ks = torch.full((4,), hint, dtype=torch.int32, device="cuda")
        expect_not_implemented(lambda: func(logits, ks, None, top_k_min=hint, top_k_max=hint))
        rows.append(dict(uniform_k=hint, supported=False, valid=True))
    expect_not_implemented(lambda: func(logits, None, p, top_k_min=20, top_k_max=20))
    rows.append(dict(top_k=None, uniform_k=20, supported=False, valid=True))
    for vocab in (4096, 32768, 65535):
        small = torch.empty((4, vocab), device="cuda")
        expect_not_implemented(lambda: func(small, k, p, top_k_min=20, top_k_max=20))
        rows.append(dict(vocab=vocab, supported=False, valid=True))
    wrong_shape = torch.full((3,), 20, dtype=torch.int32, device="cuda")
    expect_not_implemented(lambda: func(logits, wrong_shape, p, top_k_min=20, top_k_max=20))
    rows.append(dict(k_shape=[3], supported=False, valid=True))
    for bound in (129, 256):
        ks = torch.full((4,), bound, dtype=torch.int32, device="cuda")
        actual = func(logits.clone(), ks, None, top_k_min=bound, top_k_max=bound)
        assert torch.equal(torch.isfinite(actual).sum(-1), ks.long())
        rows.append(dict(uniform_k=bound, supported=True, valid=True))
    ks = torch.tensor([1, 20, 64, 100], dtype=torch.int32, device="cuda")
    ps = torch.tensor([0.1, 0.5, 0.8, 1.0], device="cuda")
    kwargs = module["prepare_kwargs"]("flashinfer-fastpath", ks, ps, logits.shape[-1])
    assert kwargs == {"top_k_min": 1, "top_k_max": 100}
    for bound, p_values in itertools.product((100, 128), (None, ps)):
        actual = func(logits.clone(), ks, p_values, top_k_min=1, top_k_max=bound).softmax(-1)
        reference = module["apply_top_k_top_p_pytorch"](logits.clone(), ks, p_values).softmax(-1)
        assert torch.allclose(actual, reference, rtol=1e-5, atol=1e-6)
        rows.append(dict(top_k=[1, 20, 64, 100], top_k_max=bound, top_p=p_values is not None, supported=True, valid=True))
    return rows


def expect_not_implemented(call) -> None:
    try:
        call()
    except NotImplementedError:
        return
    raise AssertionError("Expected an unsupported fast-path configuration to raise")


def check(module: dict, distribution: str, batch: int, vocab: int, k: int, p: float | None) -> dict:
    torch.manual_seed(42)
    logits = module["create_logits"](batch, vocab, distribution=distribution)
    ks = torch.full((batch,), k, dtype=torch.int32, device="cuda")
    ps = torch.full((batch,), p, device="cuda") if p is not None else None
    kwargs = module["prepare_kwargs"]("flashinfer-fastpath", ks, ps, vocab)
    assert kwargs == {"top_k_min": k, "top_k_max": k}
    actual = module["apply_top_k_top_p_flashinfer_fastpath"](logits.clone(), ks, ps, **kwargs)
    reference = module["apply_top_k_top_p_pytorch"](logits.clone(), ks, ps)
    mask = torch.isfinite(actual)
    count = mask.sum(-1)
    values = logits.topk(k, sorted=True).values.double()
    prefix = values.softmax(-1).cumsum(-1)
    last = (count - 1).clamp(0, k - 1).unsqueeze(-1)
    mass = prefix.gather(1, last).squeeze(-1)
    previous = prefix.gather(1, (last - 1).clamp_min(0)).squeeze(-1)
    previous = torch.where(count == 1, 0, previous)
    cutoff = float(ps[0]) if ps is not None else 1.0
    nucleus = (mass >= cutoff - 1e-7) & (previous <= cutoff + 1e-7)
    if p is None or p == 1:
        nucleus = count == k
    ranking = torch.where(mask, logits, float("inf")).amin(-1) >= torch.where(mask, -float("inf"), logits).amax(-1)
    basic = actual.shape == logits.shape and actual.dtype == logits.dtype and not bool(actual.isnan().any())
    basic = basic and torch.equal(actual[mask], logits[mask]) and bool(((count >= 1) & (count <= k)).all())
    valid = bool(basic and ranking.all() and nucleus.all())
    row = dict(distribution=distribution, batch=batch, vocab=vocab, top_k=k, top_p=p,
               valid=valid, ranking_valid=bool(ranking.all()), nucleus_valid=bool(nucleus.all()),
               reference_exact=bool(torch.equal(actual, reference)))
    print(json.dumps(row), flush=True)
    return row


if __name__ == "__main__":
    main()
