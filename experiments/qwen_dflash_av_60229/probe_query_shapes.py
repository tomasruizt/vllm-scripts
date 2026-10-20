"""Exact-PR target-forward CUDA graphs, independent of startup graph padding."""

import json
import os
from dataclasses import fields, is_dataclass
from pathlib import Path

import torch


def main() -> None:
    from vllm import LLM

    llm = LLM(
        model=os.environ["MODEL"], language_model_only=True,
        max_model_len=16384, max_num_seqs=512, max_num_batched_tokens=16384,
        gpu_memory_utilization=0.92, enable_prefix_caching=False,
        worker_extension_cls="probe_query_shapes.QueryShapeWorker",
        speculative_config=dict(
            method="dflash", model="z-lab/Qwen3.5-35B-A3B-DFlash",
            revision="52cb554b4995dede3e2e1bdb129cdb1f3529332b",
            num_speculative_tokens=7, draft_sample_method="probabilistic",
            enable_adaptive_verification=True,
        ),
    )
    print(llm.collective_rpc("probe_query_shapes"), flush=True)


class QueryShapeWorker:
    def probe_query_shapes(self) -> str:
        runner = self.model_runner
        output = Path(os.environ["PROBE_OUTPUT"])
        output.mkdir(parents=True, exist_ok=True)
        with runner.step_timing.collect() as startup:
            for _ in range(3):
                for tokens in (128, 256, 512, 1024):
                    runner._dummy_run(tokens, context_len=1024)
        (output / "startup.json").write_text(json.dumps([describe(sample) for sample in startup], indent=2))
        rows = []
        for valid_states in (False, True):
            for requests, query in ((128, 1), (128, 2), (128, 8), (512, 1), (512, 2)):
                row = probe_shape(runner, output, requests, query, valid_states)
                rows.append(row)
                (output / "shapes.json").write_text(json.dumps(rows, indent=2))
                print(json.dumps(row), flush=True)
        (output / "DONE").write_text("ok\n")
        return str(output)


@torch.inference_mode()
def probe_shape(runner: object, output: Path, requests: int, query: int, valid: bool) -> dict:
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import set_forward_context
    from vllm.v1.kv_cache_interface import MambaSpec
    from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
    from vllm.v1.worker.gpu.input_batch import InputBatch, set_dummy_context

    tokens = requests * query
    label = f"r{requests}-q{query}-valid{int(valid)}"
    batch = InputBatch.make_dummy(requests, tokens, runner.input_buffers, max_query_len=query)
    tables, slots = runner.prepare_dummy_attn(batch)
    set_dummy_context(batch, runner.block_tables, 1024, runner.kv_cache_config.num_blocks,
                      runner.max_model_len, input_block_tables=tables)
    allocations = []
    for table, group in zip(tables, runner.kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        if not isinstance(spec, MambaSpec):
            continue
        if valid:
            width = query
            assert table.shape[1] >= width
            table.zero_()
            indices = torch.arange(1, requests * width + 1, dtype=table.dtype, device=table.device)
            assert indices[-1].item() < runner.kv_cache_config.num_blocks
            table[:, :width].copy_(indices.view(requests, width))
        allocations.append(dict(shape=list(table.shape), min=int(table.min()), max=int(table.max())))
    metadata = runner.model_state.prepare_attn(
        batch, CUDAGraphMode.NONE, tables, slots, runner.attn_groups,
        runner.kv_cache_config, for_capture=True,
    )
    mapping = build_slot_mappings_by_layer(slots, runner.kv_cache_config)
    inputs = {"input_ids": batch.input_ids, "positions": batch.positions, "intermediate_tensors": None,
              **runner.model_state.prepare_dummy_inputs(requests, tokens)}
    inputs["positions"].copy_(batch.positions.expand_as(inputs["positions"]))
    summary = {key: describe(value) for key, value in metadata.items()}
    (output / f"{label}-metadata.json").write_text(json.dumps(summary, indent=2))

    def forward() -> object:
        with set_forward_context(metadata, runner.vllm_config, num_tokens=tokens,
                                 cudagraph_runtime_mode=CUDAGraphMode.NONE,
                                 slot_mapping=mapping, is_padding=batch.is_padding):
            return runner.model(**inputs)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        hidden = forward()
    times = []
    for _ in range(3):
        for _ in range(5):
            graph.replay()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(50):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 50)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(output / f"{label}-trace.json"))
    del hidden
    return dict(label=label, requests=requests, query=query, context=1024,
                valid_states=valid, graph_mode="FULL-exact", target_ms=times,
                mamba_tables=allocations)


def describe(value: object) -> object:
    if isinstance(value, torch.Tensor):
        result = dict(shape=list(value.shape), dtype=str(value.dtype), stride=list(value.stride()))
        if value.numel():
            result.update(min=value.min().item(), max=value.max().item(),
                          head=value.flatten()[:16].tolist())
        return result
    if is_dataclass(value):
        return {field.name: describe(getattr(value, field.name)) for field in fields(value)}
    if hasattr(value, "_asdict"):
        return {key: describe(item) for key, item in value._asdict().items()}
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


if __name__ == "__main__":
    main()
