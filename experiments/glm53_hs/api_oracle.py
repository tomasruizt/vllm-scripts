"""Observe graph replay outputs before extraction storage, outside captured code."""

import hashlib
import inspect
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


class ApiOracleExtension:
    def install_api_oracle(self, directory):
        from vllm.distributed import get_tensor_model_parallel_rank

        if get_tensor_model_parallel_rank() != 0:
            return {"observing": False}
        self._api_oracle = ApiOracle(self.model_runner, directory)
        return self._api_oracle.status()

    def api_oracle_status(self, enabled="false"):
        if not hasattr(self, "_api_oracle"):
            return {"observing": False}
        self._api_oracle.enabled = enabled == "true"
        return self._api_oracle.status()


class ApiOracle:
    def __init__(self, runner, directory):
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

        self.runner = runner
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.enabled = False
        self.replays = 0
        self.step = 0
        self.start_replays = 0
        self.target_replays = 0
        self.storage_replays = 0
        self.in_breakable = False
        replay = torch.cuda.CUDAGraph.replay

        def counted_replay(graph):
            result = replay(graph)
            if not self.in_breakable:
                self.replays += 1
            return result

        torch.cuda.CUDAGraph.replay = counted_replay
        breakable_replay = BreakableCUDAGraphCapture.replay

        def counted_breakable_replay(capture):
            # Breakable captures retain bound replay methods from initialization.
            self.in_breakable = True
            try:
                result = breakable_replay(capture)
            finally:
                self.in_breakable = False
            self.replays += capture.num_graphs
            return result

        BreakableCUDAGraphCapture.replay = counted_breakable_replay
        execute = runner.execute_model

        def execute_model(*args, **kwargs):
            self.start_replays = self.replays
            return execute(*args, **kwargs)

        runner.execute_model = execute_model
        proposer = getattr(runner, "speculator", None)
        if proposer is None:
            proposer = runner.drafter
        propose = proposer.propose
        signature = inspect.signature(propose)

        def observed_propose(*args, **kwargs):
            arguments = signature.bind(*args, **kwargs).arguments
            target_replays = self.replays - self.start_replays
            self.target_replays += target_replays
            if self.enabled:
                self.save_reference(arguments, target_replays)
            before = self.replays
            result = propose(*args, **kwargs)
            self.storage_replays += self.replays - before
            return result

        proposer.propose = observed_propose

    def status(self):
        return {
            "observing": True,
            "references_enabled": self.enabled,
            "target_replays": self.target_replays,
            "storage_replays": self.storage_replays,
            "reference_steps": self.step,
            "graph_mode": str(self.runner.compilation_config.cudagraph_mode),
        }

    def save_reference(self, arguments, replays):
        if "input_batch" in arguments:
            batch = arguments["input_batch"]
            req_ids = list(batch.req_ids)
            boundaries = batch.query_start_loc_np[: len(req_ids) + 1].copy()
            ids, positions = batch.input_ids, batch.positions
            auxiliary = arguments["aux_hidden_states"]
        else:
            req_ids = list(self.runner.input_batch.req_ids)
            boundaries = self.runner.query_start_loc.np[: len(req_ids) + 1].copy()
            ids, positions = self.runner.input_ids.gpu, self.runner.positions
            auxiliary = arguments["target_hidden_states"]
        total = int(boundaries[-1])
        ids = ids[:total].detach().cpu().long().clone()
        positions = positions[:total].detach().cpu().long().clone()
        states = torch.stack([value[:total].detach().cpu().clone() for value in auxiliary], dim=1)
        for i, req_id in enumerate(req_ids):
            start, stop = map(int, boundaries[i : i + 2])
            name = hashlib.sha256(req_id.encode()).hexdigest()[:20]
            filename = f"{self.step:05d}-{name}.safetensors"
            save_file(
                {"token_ids": ids[start:stop].contiguous(),
                 "positions": positions[start:stop].contiguous(),
                 "hidden_states": states[start:stop].contiguous()},
                self.directory / filename,
            )
            with (self.directory / "manifest.jsonl").open("a") as stream:
                stream.write(json.dumps({"request_id": req_id, "file": filename,
                                         "step": self.step, "target_replays": replays}) + "\n")
        self.step += 1
