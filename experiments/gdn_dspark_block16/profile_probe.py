"""Diagnostic-only c1 timings, with device results copied after requests drain."""

import functools
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

STATE = {"enabled": False, "rows": [], "current": None, "buffer": None}


class C1ProfileProbe:
    def init_worker_extension(self) -> None:
        from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
            AdaptiveVerificationManager,
        )
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

        wrap(GPUModelRunner, "execute_model", execute)
        wrap(GPUModelRunner, "prepare_inputs", prepared)
        wrap(GPUModelRunner, "sample", sampled)
        wrap(GPUModelRunner, "sample_tokens", finished)
        wrap(ModelCudaGraphManager, "run_fullgraph", stage("target"))
        wrap(DFlashSpeculator, "propose", stage("drafter"))
        wrap(AdaptiveVerificationManager, "record_confidences", stage("confidence"))

    def begin_c1_profile(self, label: str = "trace") -> dict:
        STATE["rows"] = []
        STATE["current"] = None
        STATE["label"] = label
        STATE["buffer"] = torch.full(
            (20000, 3), -1, dtype=torch.int32, device=self.device
        )
        STATE["enabled"] = True
        return {"rank": dist.get_rank(), "started": True}

    def end_c1_profile(self) -> dict:
        from vllm.utils.gpu_sync_debug import gpu_sync_allowed

        STATE["enabled"] = False
        rows = STATE["rows"]
        with gpu_sync_allowed():
            torch.cuda.synchronize()
            values = STATE["buffer"][: len(rows)].cpu().tolist()
        result = []
        previous = None
        for row, counts in zip(rows, values):
            output = {key: value for key, value in row.items() if key != "events"}
            output.update(
                gpu_query_len=counts[0],
                sampled_tokens=counts[1],
                rejected_tokens=counts[2],
            )
            for name, pair in row["events"].items():
                if len(pair) == 2:
                    output[f"{name}_gpu_ms"] = pair[0].elapsed_time(pair[1])
            if (
                previous is not None
                and previous.get("request_id") == row.get("request_id")
                and len(previous["events"]["step"]) == 2
            ):
                output["inter_step_gpu_gap_ms"] = previous["events"]["step"][
                    1
                ].elapsed_time(row["events"]["step"][0])
                output["inter_step_cpu_gap_ms"] = (
                    row["cpu_start_ns"] - previous["cpu_end_ns"]
                ) / 1e6
            result.append(output)
            previous = row
        path = (
            Path(os.environ["C1_PROFILE_OUTPUT"])
            / f"steps-{STATE['label']}-rank{dist.get_rank()}.jsonl"
        )
        path.write_text("".join(json.dumps(row) + "\n" for row in result))
        STATE["buffer"] = None
        STATE["rows"] = []
        return {"rank": dist.get_rank(), "steps": len(result)}


def execute(original, obj, *args, **kwargs):
    index = len(STATE["rows"])
    assert index < 20000
    row = {
        "step": index,
        "rank": dist.get_rank(),
        "events": {},
        "cpu_start_ns": time.perf_counter_ns(),
    }
    row["events"]["step"] = [event()]
    STATE["rows"].append(row)
    STATE["current"] = row
    return timed("execute", original, obj, *args, **kwargs)


def prepared(original, obj, *args, **kwargs):
    batch = timed("prepare", original, obj, *args, **kwargs)
    row = STATE["current"]
    row.update(
        num_reqs=int(batch.num_reqs),
        target_tokens=int(batch.num_tokens),
        padded_tokens=int(batch.num_tokens_after_padding),
        has_prefill=bool(batch.has_prefill),
        request_id=batch.req_ids[0],
        prefill_tokens=int(batch.prefill_len_np[0]),
        context_tokens=int(batch.num_computed_tokens_np[0]),
    )
    if batch.num_reqs == 1:
        STATE["buffer"][row["step"], 0].copy_(batch.query_start_loc[1])
    return batch


def sampled(original, obj, *args, **kwargs):
    result = timed("sample", original, obj, *args, **kwargs)
    row = STATE["current"]
    if row.get("num_reqs") == 1:
        STATE["buffer"][row["step"], 1].copy_(result[1][0])
        STATE["buffer"][row["step"], 2].copy_(result[2][0])
    return result


def finished(original, obj, *args, **kwargs):
    result = timed("sample_tokens", original, obj, *args, **kwargs)
    row = STATE["current"]
    row["events"]["step"].append(event())
    row["cpu_end_ns"] = time.perf_counter_ns()
    return result


def stage(name: str):
    def call(original, obj, *args, **kwargs):
        return timed(name, original, obj, *args, **kwargs)

    return call


def timed(name: str, original, obj, *args, **kwargs):
    row = STATE["current"]
    pair = [event()]
    started = time.perf_counter_ns()
    torch.cuda.nvtx.range_push(f"c1/{name}/step={row['step']}")
    try:
        return original(obj, *args, **kwargs)
    finally:
        torch.cuda.nvtx.range_pop()
        pair.append(event())
        row["events"][name] = pair
        row[f"{name}_cpu_ms"] = (time.perf_counter_ns() - started) / 1e6


def event() -> torch.cuda.Event:
    result = torch.cuda.Event(enable_timing=True)
    result.record()
    return result


def wrap(cls, name: str, hook) -> None:
    original = getattr(cls, name)

    @functools.wraps(original)
    def call(obj, *args, **kwargs):
        if not STATE["enabled"]:
            return original(obj, *args, **kwargs)
        return hook(original, obj, *args, **kwargs)

    setattr(cls, name, call)
