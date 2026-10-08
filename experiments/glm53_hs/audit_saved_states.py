"""Audit saved extraction results on CPU without loading the model."""

import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file


def main():
    torch.set_num_threads(2)
    root = Path(os.environ.get("HS_RUNS_DIR", Path.home() / "benchmarks/glm-hs-validation"))
    result = {}
    for name in ("runner0", "diagnostic-runner0", "diagnostic-runner1"):
        folder = root / name
        files = sorted(folder.glob("*.safetensors"), key=request_id)
        expected = 6 if name == "runner0" else 15
        assert len(files) == expected, (name, len(files))
        states = [read_checked(path, (8, 881, 1921)[i % 3]) for i, path in enumerate(files)]
        for i, state in enumerate(states):
            assert torch.equal(state["token_ids"], states[i % 3]["token_ids"])
        groups = (
            [("batch0_vs_serial", 3, 0)] if expected == 6 else [
                ("serial_repeat", 3, 0),
                ("batch0_vs_serial", 6, 0),
                ("batch1_vs_serial", 9, 0),
                ("batch_repeat", 9, 6),
                ("serial_after_batch", 12, 0),
            ]
        )
        rows = []
        for kind, actual_start, reference_start in groups:
            for request in range(3):
                a, r = actual_start + request, reference_start + request
                row = compare(states[a]["hidden_states"], states[r]["hidden_states"])
                rows.append({"kind": kind, "request": request, "actual": files[a].name,
                             "reference": files[r].name, **row})
        result[name] = {"valid_artifacts": len(files), "comparisons": rows}
        print(name, len(files), "valid artifacts;", sum(row["close"] for row in rows),
              "/", len(rows), "comparisons pass")
    (root / "audit.json").write_text(json.dumps(result, indent=2) + "\n")


def request_id(path):
    return int(path.name.split("-", 1)[0])


def read_checked(path, tokens):
    obj = load_file(path)
    states = obj["hidden_states"]
    assert states.shape == (tokens, 3, 4096)
    assert obj["token_ids"].shape == (tokens,)
    assert states.dtype == torch.bfloat16
    assert torch.isfinite(states).all()
    assert torch.count_nonzero(states) > 0
    return obj


def compare(actual, reference):
    close = True
    try:
        torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-2)
    except AssertionError:
        close = False
    delta = (actual.float() - reference.float()).abs()
    return {
        "shape": list(actual.shape), "close": close,
        "bitwise_equal": torch.equal(actual, reference),
        "max_abs": delta.max().item(), "mean_abs": delta.mean().item(),
        "layer_max_abs": delta.amax(dim=(0, 2)).tolist(),
    }


if __name__ == "__main__":
    main()
