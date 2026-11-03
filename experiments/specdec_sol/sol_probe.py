"""Worker extension: NVTX ranges around spec-decode phases, and one NVTX mark per step with its batch shape.

The wrapped methods are found on the live runner, so the probe works with Model Runner V1 and V2 and any speculator.
Ranges are labeled "sol/<Class>.<method>", they cost ~1 us each and are recorded only inside an nsys capture range.
"""

import functools
import math
import re

import torch

STEP_METHOD = "execute_model"  # one call per engine step, sol.py splits steps on it
RUNNER_METHODS = [STEP_METHOD, "sample_tokens", "prepare_inputs", "_prepare_inputs", "sample", "_sample",
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
        cache = cache_layers(runner.get_kv_cache_spec(), {name for name, _ in target.named_modules()})
        hidden = getattr(text_config, "hidden_size", 0)
        return {**info, "top_k": top_k, "gemm_flops_per_s": gemm_rate(), "layers": layers, "cache": cache, "hidden_size": hidden}


TOP_K_KEYS = ["num_experts_per_tok", "moe_top_k", "num_experts_per_token", "top_k"]
EXPERT_PARAM = re.compile(r"experts|\.w13_|\.w2_|\.w1_|\.w3_")


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
            result.append({"name": name, "kind": kind, "bytes": sum(p.nbytes for p in params),
                           "elems": sum(p.numel() for p in params)})  # fmt: skip
    return result


def cache_layers(spec: dict, target_modules: set[str]) -> list[dict]:
    """Attention layers with their KV bytes per token, recurrent layers with their conv and SSM state bytes."""
    result = []
    for name, layer in spec.items():
        owner = "target" if name in target_modules else "draft"
        if hasattr(layer, "shapes"):
            sizes = [math.prod(s) * torch.empty(0, dtype=d).element_size() for s, d in zip(layer.shapes, layer.dtypes)]
            result.append({"name": name, "owner": owner, "kind": "state", "conv_bytes": sizes[0], "ssm_bytes": sum(sizes[1:])})
        else:
            page = getattr(layer, "real_page_size_bytes", layer.page_size_bytes)
            result.append({"name": name, "owner": owner, "kind": "attention", "kv_bytes_per_token": page / layer.block_size})
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
        if mark:
            step_mark(args[0] if args else kwargs.get("scheduler_output"))
        torch.cuda.nvtx.range_push(f"sol/{label}")
        try:
            return original(obj, *args, **kwargs)
        finally:
            torch.cuda.nvtx.range_pop()

    call._sol_wrapped = True
    setattr(cls, name, call)
    return label


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
