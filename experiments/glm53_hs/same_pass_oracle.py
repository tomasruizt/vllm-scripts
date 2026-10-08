"""Same-forward-pass oracle, independent of KV cache addressing and extraction."""

import hashlib
import json
from functools import partial
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


class OracleWorkerExtension:
    def install_same_pass_oracle(self, directory, layer_ids):
        return install_oracle(self, directory, layer_ids)


def install_oracle(worker, directory, layer_ids):
    """Install eager-only observation hooks after engine initialization."""
    from vllm.distributed import get_tensor_model_parallel_rank

    if get_tensor_model_parallel_rank() != 0:
        return {"rank": get_tensor_model_parallel_rank(), "observing": False}
    oracle = SamePassOracle(worker.model_runner, directory, layer_ids)
    worker._same_pass_oracle = oracle
    return {"rank": 0, "observing": True, "layers": oracle.layer_ids}


def check_saved_output(output, directory, layer_ids, actual=None):
    """Compare the user's artifact to independently assembled forward fragments."""
    from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
        load_hidden_states,
    )

    saved_path = Path(output.kv_transfer_params["hidden_states_path"])
    req_id = saved_path.stem
    manifest = Path(directory) / "manifest.jsonl"
    entries = [json.loads(line) for line in manifest.read_text().splitlines()]
    entries = [entry for entry in entries if entry["request_id"] == req_id]
    assert entries, f"No forward reference for {req_id}"
    pieces = [load_file(Path(directory) / entry["file"]) for entry in entries]
    order = torch.cat([piece["positions"] for piece in pieces]).argsort()
    positions = torch.cat([piece["positions"] for piece in pieces])[order]
    token_ids = torch.cat([piece["token_ids"] for piece in pieces])[order]
    reference = torch.cat([piece["hidden_states"] for piece in pieces])[order]
    expected_ids = torch.tensor(output.prompt_token_ids)
    assert torch.equal(
        positions, torch.arange(len(expected_ids))
    ), "Missing/duplicate positions"
    assert torch.equal(
        token_ids, expected_ids
    ), "Forward token/request mapping is wrong"
    if actual is None:
        actual = load_hidden_states(str(saved_path))
    assert torch.equal(actual["token_ids"], expected_ids)
    assert actual["hidden_states"].shape == (
        len(expected_ids),
        len(layer_ids),
        reference.shape[-1],
    )
    assert actual["hidden_states"].dtype == reference.dtype
    assert torch.isfinite(actual["hidden_states"]).all()
    assert torch.count_nonzero(actual["hidden_states"]) > 0
    assert_exact(actual["hidden_states"], reference, req_id)
    assert torch.equal(
        actual["hidden_states"].contiguous().view(torch.uint8),
        reference.contiguous().view(torch.uint8),
    ), f"{req_id}: byte mismatch"
    row = {
        "request_id": req_id,
        "shape": list(reference.shape),
        "chunks": len(pieces),
        "bitwise_equal": True,
        "layer_ids": sorted(layer_ids),
        "path": str(saved_path),
    }
    print(json.dumps(row), flush=True)
    return row, reference


class SamePassOracle:
    def __init__(self, runner, directory, layer_ids):
        self.runner = runner
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.layer_ids = sorted(layer_ids)
        self.step = 0
        self.current_batch = None
        self.expected_layers = {}
        self.pending_post = {}
        model = runner.get_model()
        language_model = getattr(model, "language_model", model)
        decoder = language_model.model
        assert (
            not decoder.is_sequence_parallel
        ), "This oracle requires replicated token rows"
        assert list(decoder.aux_hidden_state_layers) == self.layer_ids
        assert all(0 <= layer < len(decoder.layers) for layer in self.layer_ids)
        if hasattr(runner, "prepare_inputs"):
            original = runner.prepare_inputs

            def prepare_inputs(*args, **kwargs):
                batch = original(*args, **kwargs)
                self.current_batch = batch
                return batch

            runner.prepare_inputs = prepare_inputs
        model.register_forward_pre_hook(self.begin_forward, with_kwargs=True)
        for index in self.layer_ids:
            layer = decoder.layers[index]
            layer.register_forward_pre_hook(partial(self.enter_layer, index))
            if hasattr(layer, "mhc_fused_post_pre_op"):
                layer.mhc_fused_post_pre_op.register_forward_hook(
                    partial(self.completed_residual, index)
                )
        model.register_forward_hook(self.end_forward, with_kwargs=True)

    def begin_forward(self, module, args, kwargs):
        self.expected_layers = {}
        self.pending_post = {}
        if self.current_batch is not None:
            batch = self.current_batch
            self.req_ids = list(batch.req_ids)
            self.boundaries = batch.query_start_loc_np[: len(self.req_ids) + 1].copy()
            ids = batch.input_ids
            positions = batch.positions
        else:
            self.req_ids = list(self.runner.input_batch.req_ids)
            self.boundaries = self.runner.query_start_loc.np[
                : len(self.req_ids) + 1
            ].copy()
            ids = self.runner.input_ids.gpu
            positions = self.runner.positions
        total = int(self.boundaries[-1])
        self.token_ids = ids[:total].detach().cpu().long().clone()
        self.positions = positions[:total].detach().cpu().long().clone()

    def enter_layer(self, index, module, args):
        _, hidden, _, post, _ = args
        if post is None:
            self.expected_layers[index] = hidden.detach().cpu().clone()
        else:
            self.pending_post[index] = True

    def completed_residual(self, index, module, args, output):
        if not self.pending_post.pop(index, False):
            return
        # Normal model execution materializes the incoming residual in its
        # fused post/pre operation. Do not call the auxiliary capture helper.
        residual = output[0].detach().cpu()
        self.expected_layers[index] = residual.float().mean(dim=1).to(residual.dtype)

    def end_forward(self, module, args, kwargs, output):
        _, aux = output
        assert len(aux) == len(self.layer_ids)
        assert set(self.expected_layers) == set(
            self.layer_ids
        ), "Missing layer observation"
        direct = torch.stack([value.detach().cpu().clone() for value in aux], dim=1)
        for slot, index in enumerate(self.layer_ids):
            assert_exact(
                direct[:, slot], self.expected_layers[index], f"layer {index} semantics"
            )
        for i, req_id in enumerate(self.req_ids):
            start, stop = map(int, self.boundaries[i : i + 2])
            name = hashlib.sha256(req_id.encode()).hexdigest()[:20]
            filename = f"{self.step:05d}-{name}.safetensors"
            save_file(
                {
                    "hidden_states": direct[start:stop].contiguous(),
                    "token_ids": self.token_ids[start:stop].contiguous(),
                    "positions": self.positions[start:stop].contiguous(),
                },
                self.directory / filename,
            )
            with (self.directory / "manifest.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "request_id": req_id,
                            "file": filename,
                            "step": self.step,
                            "tokens": stop - start,
                            "layer_semantics_exact": True,
                        }
                    )
                    + "\n"
                )
        self.step += 1


def assert_exact(actual, reference, context):
    assert actual.shape == reference.shape, context
    assert actual.dtype == reference.dtype, context
    if not torch.equal(actual, reference):
        delta = (actual.float() - reference.float()).abs()
        raise AssertionError(
            f"{context}: {(actual != reference).sum().item()} unequal elements; "
            f"max_abs={delta.max().item()}"
        )
