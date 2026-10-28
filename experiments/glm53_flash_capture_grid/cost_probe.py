"""Time real mixed/prefill target forwards in the V2 model runner."""
from collections import Counter
import json
from pathlib import Path
import time


class CostProbeExtension:
    def capture_cost_capabilities(self) -> dict:
        runner = self.model_runner
        manager = runner.cudagraph_manager
        assert hasattr(runner, "model_state") and hasattr(manager, "run_pw_graph")
        return dict(rank=self.rank, runner_module=type(runner).__module__,
                    breakable_graphs=manager.use_breakable_cg,
                    compilation_mode=runner.compilation_config.mode.name)

    def capture_cost_probe(self, directory: str, phase: str) -> dict:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        runner = self.model_runner
        if hasattr(runner, "_capture_cost_original"):
            runner.cudagraph_manager.run_pw_graph = runner._capture_cost_original
            runner.model_state.prepare_attn = runner._capture_cost_attn_original
            CUDAGraphWrapper.__call__ = runner._capture_cost_wrapper_original
            BreakableCUDAGraphCapture.replay = runner._capture_cost_breakable_original
            runner._capture_cost_file.close()
            del runner._capture_cost_original
        if phase == "off":
            return {"phase": phase}
        self.capture_cost_capabilities()
        runner._capture_cost_original = runner.cudagraph_manager.run_pw_graph
        runner._capture_cost_attn_original = runner.model_state.prepare_attn
        runner._capture_cost_file = (Path(directory) / f"{phase}-rank{self.rank}.jsonl").open("w", buffering=1)
        runner._capture_cost_counts = Counter()
        runner._capture_cost_step = 0
        runner._capture_cost_replays = 0
        runner._capture_cost_wrapper_original = CUDAGraphWrapper.__call__
        runner._capture_cost_breakable_original = BreakableCUDAGraphCapture.replay
        CUDAGraphWrapper.__call__ = lambda wrapper, *args, **kwargs: observe_replay(runner, wrapper, *args, **kwargs)
        BreakableCUDAGraphCapture.replay = lambda capture: observe_breakable(runner, capture)
        runner.model_state.prepare_attn = lambda batch, *args, **kwargs: prepare_attn(runner, batch, *args, **kwargs)
        runner.cudagraph_manager.run_pw_graph = lambda *args, **kwargs: timed_forward(runner, phase, *args, **kwargs)
        return {"phase": phase, "rank": self.rank}


def timed_forward(runner: object, phase: str, *args: object, **kwargs: object) -> object:
    import numpy as np
    import torch
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context
    from vllm.utils.gpu_sync_debug import gpu_sync_allowed

    context = get_forward_context()
    original_mode = context.cudagraph_runtime_mode
    descriptor = context.batch_descriptor
    batch = runner._capture_cost_batch
    num_reqs = batch.num_reqs
    query_lens = np.asarray(batch.num_scheduled_tokens[:num_reqs]).copy()
    computed = batch.num_computed_tokens_np[:num_reqs].copy()
    prefills = batch.is_prefilling_np[:num_reqs].copy()
    padded_tokens = batch.num_tokens_after_padding
    assert descriptor is not None and descriptor.num_tokens == padded_tokens
    assert original_mode == CUDAGraphMode.PIECEWISE
    bucket = (original_mode.name, padded_tokens, bool(prefills.any()))
    count = runner._capture_cost_counts[bucket]
    runner._capture_cost_counts[bucket] += 1
    bypass = original_mode != CUDAGraphMode.NONE and count % 4 in (1, 2)
    mode = "bypass" if bypass else "graph" if original_mode != CUDAGraphMode.NONE else "natural_eager"
    assert not context.skip_compiled, "The probe must retain compilation"
    if bypass:
        context.cudagraph_runtime_mode = CUDAGraphMode.NONE
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    runner._capture_cost_replays = 0
    try:
        with gpu_sync_allowed():
            torch.cuda.synchronize()
            wall_start = time.perf_counter_ns()
            start.record()
            output = runner._capture_cost_original(*args, **kwargs)
            end.record()
            end.synchronize()
            wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
    finally:
        context.cudagraph_runtime_mode = original_mode
    replays = runner._capture_cost_replays
    assert not bypass or replays == 0, "Bypassed forwards must not replay graphs"
    assert original_mode == CUDAGraphMode.NONE or bypass or replays > 0, "Expected captured graph replay"
    row = dict(phase=phase, step=runner._capture_cost_step, mode=mode,
               dispatched_mode=original_mode.name, warmup=count < 8,
               query_tokens=int(batch.num_tokens), padded_tokens=padded_tokens,
               num_reqs=num_reqs, prefill_reqs=int(prefills.sum()),
               prefill_tokens=int(query_lens[prefills].sum()),
               context_mean=float(computed.mean()), context_max=int(computed.max()),
               query_lens=query_lens.tolist(), graph_replays=replays,
               gpu_ms=start.elapsed_time(end), wall_ms=wall_ms)
    runner._capture_cost_step += 1
    runner._capture_cost_file.write(json.dumps(row) + "\n")
    return output


def prepare_attn(runner: object, batch: object, *args: object, **kwargs: object) -> object:
    runner._capture_cost_batch = batch
    return runner._capture_cost_attn_original(batch, *args, **kwargs)


def observe_breakable(runner: object, capture: object) -> object:
    runner._capture_cost_replays += capture._num_graphs
    return runner._capture_cost_breakable_original(capture)


def observe_replay(runner: object, wrapper: object, *args: object, **kwargs: object) -> object:
    from vllm.forward_context import get_forward_context
    context = get_forward_context()
    entry = wrapper.concrete_cudagraph_entries.get(context.batch_descriptor)
    if context.cudagraph_runtime_mode == wrapper.runtime_mode and entry is not None and entry.cudagraph is not None:
        runner._capture_cost_replays += 1
    return runner._capture_cost_wrapper_original(wrapper, *args, **kwargs)
