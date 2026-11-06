"""Worker extension: NVTX ranges around spec-decode phases, and one NVTX mark per step with its batch shape.

The wrapped methods are found on the live runner, so the probe works with Model Runner V1 and V2 and any speculator.
Ranges are labeled "sol/<Class>.<method>", they cost ~1 us each and are recorded only inside an nsys capture range.
"""

import contextlib
import functools
import json
import math
import os
import re

import torch

STEP_METHOD = "execute_model"  # one call per engine step, sol.py splits steps on it
RUNNER_METHODS = [STEP_METHOD, "capture_model", "sample_tokens", "prepare_inputs", "_prepare_inputs", "sample", "_sample",
                  "postprocess_sampled", "propose_draft_token_ids"]  # fmt: skip
SPECULATOR_ATTRS = ["speculator", "drafter"]  # V2, V1
SPECULATOR_METHODS = ["propose"]


class SolProbe:
    def init_worker_extension(self) -> list[str]:
        """Called once over collective_rpc after startup, returns the wrapped methods."""
        runner = self.model_runner
        targets = [(type(runner), name) for name in RUNNER_METHODS]
        speculators = [getattr(runner, attr, None) for attr in SPECULATOR_ATTRS]
        targets += [(type(s), name) for s in speculators if s is not None for name in SPECULATOR_METHODS]
        targets.append((torch.cuda.CUDAGraph, "replay"))
        return [label for cls, name in targets if (label := wrap(cls, name))]

    def sol_shape_capture_status(self) -> bool:
        return bool(getattr(self.model_runner, "_sol_shape_capture_started", False))

    def sol_start_shape_measurement(self, concurrency: int | None = None) -> None:
        suffix = f" c={concurrency}" if concurrency is not None else ""
        torch.cuda.nvtx.mark("sol/measurement_start" + suffix)

    def sol_stop_shape_measurement(self, concurrency: int | None = None, stop: bool = True) -> None:
        suffix = f" c={concurrency}" if concurrency is not None else ""
        torch.cuda.nvtx.mark("sol/measurement_end" + suffix)
        if stop:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()  # Every rank must emit its end marker before collection stops.
            torch.cuda.profiler.stop()

    def sol_replay_shapes(self, shapes: list[dict], profile: bool = False) -> int:
        """Run the runner's dummy forward (target and draft) once per recorded step shape, so a profiler that cannot
        reproduce the live workload's shapes (e.g. adaptive verification choosing differently under ncu) still
        launches their kernels. `shapes`: dicts with reqs, tokens (executed) and context (per request).

        With `profile`, every shape is first run unprofiled (autotuning, compilation), then run again inside one
        cudaProfilerStart/Stop window, so ncu (profile-from-start off, no launch filter) records every launch of one
        step per shape. spin_kernel launches mark where each shape begins, ncu has no other way to tell them apart."""
        for shape in shapes:
            self.replay_shape(shape)
        if not profile:
            return len(shapes)
        torch.cuda.profiler.start()
        for shape in shapes:
            torch.cuda._sleep(1000)  # sentinel
            self.replay_shape(shape)
        torch.cuda._sleep(1000)
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        return len(shapes)

    def replay_shape(self, shape: dict) -> None:
        runner = self.model_runner
        with request_cap(runner, shape["reqs"]), torch.inference_mode():
            # Random ids and is_profile route the tokens over the experts (zero ids or padding tokens would make MoE
            # kernels read a handful of experts), valid_dummy_state_slots keeps the GDN layers on their real kernels.
            _, sample_hidden = runner._dummy_run(shape["tokens"], context_len=shape.get("context", 0), is_profile=True,
                                                 valid_dummy_state_slots=True, randomize_inputs=True)
            if sample_hidden is not None:
                runner._dummy_sampler_run(sample_hidden)  # LM head and sampling, like profile_run() does after the forward
        torch.cuda.synchronize()

    def sol_model_info(self) -> dict:
        """What the analytic SOL needs: bytes and elements of dense and expert weights, KV bytes per token and
        recurrent state bytes per request, for target and draft, plus the measured BF16 GEMM rate of this GPU."""
        runner = self.model_runner
        target = runner.get_model()
        draft = runner.get_draft_model() if hasattr(runner, "get_draft_model") else None
        target_ptrs = {p.data_ptr() for p in target.parameters()}
        text_config = runner.model_config.hf_text_config
        top_k = next((getattr(text_config, k) for k in TOP_K_KEYS if getattr(text_config, k, None)), None)
        info = {"target": weights(target, set()), "draft": weights(draft, target_ptrs) if draft else None}
        info.update(state_sizes(runner.get_kv_cache_spec(), {name for name, _ in target.named_modules()}))
        layers = {"target": layer_list(target, set()), "draft": layer_list(draft, target_ptrs) if draft else []}
        modules = dict(target.named_modules())
        if draft is not None:
            modules.update(draft.named_modules())
        cache = cache_layers(runner.get_kv_cache_spec(), {name for name, _ in target.named_modules()}, modules)
        hidden = getattr(text_config, "hidden_size", 0)
        dtype = runner.model_config.dtype
        return {**info, "top_k": top_k, "gemm_flops_per_s": gemm_rate(), "gemm_dtype": "bf16",
                "activation_dtype": dtype_name(dtype), "activation_bytes": torch.empty((), dtype=dtype).element_size(),
                "layers": layers, "cache": cache, "hidden_size": hidden}


TOP_K_KEYS = ["num_experts_per_tok", "moe_top_k", "num_experts_per_token", "top_k"]
EXPERT_PARAM = re.compile(r"experts|\.w13_|\.w2_|\.w1_|\.w3_")


@contextlib.contextmanager
def request_cap(runner, reqs: int):
    """_dummy_run spreads its tokens over min(tokens, max_num_reqs) requests, so capping max_num_reqs picks the request count."""
    saved = runner.max_num_reqs
    runner.max_num_reqs = min(saved, reqs)
    try:
        yield
    finally:
        runner.max_num_reqs = saved


def weights(model: torch.nn.Module, skip: set[int]) -> dict:
    """Unique parameters by storage. Expert tensors (3D, first dim = experts) are split out, since a decode step reads
    only the experts it routes to. Input-embedding tables are left out, a forward reads only a few rows."""
    params = {p.data_ptr(): (name, p) for name, p in model.named_parameters() if p.data_ptr() not in skip}
    lookup_only = embedding_ptrs(model, "VocabParallelEmbedding") - embedding_ptrs(model, "ParallelLMHead")
    params = {ptr: np for ptr, np in params.items() if ptr not in lookup_only}
    experts = {ptr: p for ptr, (name, p) in params.items() if p.dim() == 3 and EXPERT_PARAM.search(name)}
    dense = [p for ptr, (_, p) in params.items() if ptr not in experts]
    num_experts = max((p.shape[0] for p in experts.values()), default=0)
    return {
        "dense_bytes": sum(p.nbytes for p in dense), "dense_elems": sum(p.numel() for p in dense),
        "expert_bytes": sum(p.nbytes for p in experts.values()), "expert_elems": sum(p.numel() for p in experts.values()),
        "num_experts": num_experts,
    }  # fmt: skip


def embedding_ptrs(model: torch.nn.Module, type_name: str) -> set[int]:
    modules = [m for m in model.modules() if type(m).__name__ == type_name]
    return {p.data_ptr() for m in modules for p in m.parameters(recurse=False)}


def layer_list(model: torch.nn.Module, skip: set[int]) -> list[dict]:
    """GEMM-backed layers in registration order, which is the forward order in vLLM models: linear layers (one
    GEMM each), the LM head, and MoE layers (two expert GEMMs, w13 then w2). sol.py maps GEMM kernels onto them."""
    result = []
    for name, module in model.named_modules():
        params = [p for p in module.parameters(recurse=False) if p.data_ptr() not in skip]
        bases = {c.__name__ for c in type(module).__mro__}
        if hasattr(module, "w13_weight") and hasattr(module, "w2_weight"):
            w13 = module.w13_weight
            w2 = module.w2_weight
            result.append({"name": name, "kind": "moe", "experts": w13.shape[0], "w13_bytes": w13.nbytes,
                           "w2_bytes": w2.nbytes, "w13_elems": w13.numel(), "w2_elems": w2.numel(),
                           # with expert parallelism a rank holds only its local experts
                           "global_experts": getattr(module, "global_num_experts", w13.shape[0]),
                           "top_k": getattr(module, "top_k", None)})  # fmt: skip
        elif ("LinearBase" in bases or "ParallelLMHead" in bases) and params:
            kind = "lm_head" if "ParallelLMHead" in bases else "linear"
            record = {"name": name, "kind": kind, "bytes": sum(p.nbytes for p in params), "elems": sum(p.numel() for p in params)}
            inputs = getattr(module, "input_size_per_partition", getattr(module, "input_size", None))
            outputs = getattr(module, "output_size_per_partition", getattr(module, "output_size", None))
            if inputs is not None and outputs is not None:
                record.update(input_size=int(inputs), output_size=int(outputs), logical_elems=int(inputs) * int(outputs))
            record["weight_dtype"] = dtype_name(params[0].dtype)
            record["compute_dtype"] = record["weight_dtype"]  # Quantized weights require their own declared ceiling.
            result.append(record)
    return result


def cache_layers(spec: dict, target_modules: set[str], modules: dict | None = None) -> list[dict]:
    """Attention layers with their KV bytes per token, recurrent layers with their conv and SSM state bytes."""
    result = []
    for name, layer in spec.items():
        owner = "target" if name in target_modules else "draft"
        if hasattr(layer, "shapes"):
            sizes = [math.prod(s) * torch.empty(0, dtype=d).element_size() for s, d in zip(layer.shapes, layer.dtypes)]
            result.append({"name": name, "owner": owner, "kind": "state", "conv_bytes": sizes[0], "ssm_bytes": sum(sizes[1:]),
                           "state_shapes": [list(shape) for shape in layer.shapes], "state_dtypes": [dtype_name(dtype) for dtype in layer.dtypes]})
        else:
            page = getattr(layer, "real_page_size_bytes", layer.page_size_bytes)
            module = (modules or {}).get(name)
            result.append({"name": name, "owner": owner, "kind": "attention", "kv_bytes_per_token": page / layer.block_size,
                           "layout": type(layer).__name__, "kv_heads": getattr(layer, "num_kv_heads", None),
                           "query_heads": getattr(module, "num_heads", None), "head_dim": getattr(layer, "head_size", None),
                           "kv_element_bytes": torch.empty((), dtype=layer.dtype).element_size() if hasattr(layer, "dtype") else None})
    return result


def state_sizes(spec: dict, target_modules: set[str]) -> dict:
    """Attention layers read KV per context token, recurrent layers (Mamba, GDN) a fixed state per request.
    KV layer names are module paths of the model that owns the layer."""
    sizes = {f"{owner}_{kind}": 0.0 for owner in ["target", "draft"] for kind in ["kv_bytes_per_token", "state_bytes_per_req"]}
    for name, layer in spec.items():
        owner = "target" if name in target_modules else "draft"
        page = getattr(layer, "real_page_size_bytes", layer.page_size_bytes)
        if hasattr(layer, "shapes"):  # MambaSpec: one page holds a request's state
            sizes[f"{owner}_state_bytes_per_req"] += page
        else:
            sizes[f"{owner}_kv_bytes_per_token"] += page / layer.block_size
    return sizes


def gemm_rate(n: int = 8192) -> float:
    a = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
    for _ in range(3):
        a @ b
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        a @ b
    end.record()
    end.synchronize()
    return 10 * 2 * n**3 / (start.elapsed_time(end) / 1e3)


def wrap(cls: type, name: str) -> str | None:
    """Idempotent, newer vLLM calls init_worker_extension itself before the speculator exists, run.py calls it again."""
    original = getattr(cls, name, None)
    if original is None:
        return None
    owner = next(c for c in cls.__mro__ if name in vars(c))
    label = f"{owner.__name__}.{name}"
    if getattr(original, "_sol_wrapped", False):
        return label
    mark = name == STEP_METHOD

    @functools.wraps(original)
    def call(obj, *args, **kwargs):
        if name == "capture_model" and os.environ.get("SOL_NSYS_SHAPES") == "1" and not getattr(obj, "_sol_shape_capture_started", False):
            torch.cuda.profiler.start()
            obj._sol_shape_capture_started = True
        if mark:
            step_mark(args[0] if args else kwargs.get("scheduler_output"))
        torch.cuda.nvtx.range_push(f"sol/{label}")
        try:
            result = original(obj, *args, **kwargs)
            if name in ("prepare_inputs", "_prepare_inputs"):
                execution_mark(result, getattr(obj, "adaptive_verification", None) is not None)
            return result
        finally:
            torch.cuda.nvtx.range_pop()

    call._sol_wrapped = True
    setattr(cls, name, call)
    return label


def execution_mark(batch, adaptive: bool) -> None:
    """Host metadata only. Never synchronize or copy GPU query lengths into the profiling critical path."""
    if not hasattr(batch, "num_tokens") or not hasattr(batch, "num_tokens_after_padding"):
        return  # Older runners return tuples. Their execution shape remains explicitly unavailable.
    try:
        shape = prepared_shape(batch, adaptive)
    except (AttributeError, TypeError, ValueError):
        return
    torch.cuda.nvtx.mark("sol/execution " + json.dumps(shape, separators=(",", ":")))


def prepared_shape(batch, adaptive: bool) -> dict:
    tokens = int(batch.num_tokens)
    query_lengths = None
    if batch.num_reqs == 1:
        query_lengths = [tokens]
    elif not adaptive and hasattr(batch, "query_start_loc_np"):
        starts = batch.query_start_loc_np[: batch.num_reqs + 1].tolist()
        lengths = [int(b - a) for a, b in zip(starts, starts[1:])]
        if sum(lengths) == tokens:
            query_lengths = lengths
    contexts = getattr(batch, "num_computed_tokens_np", None)
    return {"actual_tokens": tokens, "padded_tokens": int(batch.num_tokens_after_padding),
            "query_lengths": query_lengths, "context_lengths": contexts[:batch.num_reqs].tolist() if contexts is not None else None,
            "logits_tokens": int(batch.logits_indices.shape[0]) if hasattr(batch, "logits_indices") else None,
            "query_lengths_source": "unavailable (GPU AV allocation)" if query_lengths is None else "host execution metadata"}


def dtype_name(dtype) -> str:
    return {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32", torch.float64: "fp64"}.get(dtype, str(dtype).removeprefix("torch."))


def step_mark(scheduler_output) -> None:
    """Never fails the step: an unknown SchedulerOutput layout only costs the batch shape (sol.py then finds no
    steady steps and says so)."""
    try:
        torch.cuda.nvtx.mark(f"sol/batch {batch_shape(scheduler_output)}")
    except Exception as error:  # noqa: BLE001
        torch.cuda.nvtx.mark(f"sol/batch unknown {type(error).__name__}")


def batch_shape(scheduler_output) -> str:
    """From the SchedulerOutput, which V1 and V2 both receive. A step is pure decode when every request schedules one
    token plus its draft tokens."""
    scheduled = scheduler_output.num_scheduled_tokens
    drafts = scheduler_output.scheduled_spec_decode_tokens
    decode = all(n == 1 + len(drafts.get(req, ())) for req, n in scheduled.items())
    cached = scheduler_output.scheduled_cached_reqs
    context = sum(cached.num_computed_tokens) + sum(r.num_computed_tokens for r in scheduler_output.scheduled_new_reqs)
    tokens = scheduler_output.total_num_scheduled_tokens
    return f"reqs={len(scheduled)} tokens={tokens} context={context} prefill={int(not decode)}"
