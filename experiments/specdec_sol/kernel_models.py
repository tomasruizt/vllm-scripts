"""Associate profiler kernels with logical layers, then delegate work and hardware pricing to benchmarking.roofline.

The execution-order mapping is inferred, not an operator graph. Its assumptions and partial metadata are retained
with every estimate. Unknown kernels never receive an activation-sized fallback. NCU references are applied separately.
"""

import itertools
import json
import re
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarking.roofline import Estimate, Hardware, estimate, unsupported  # noqa: E402
from benchmarking.nsys_shapes import estimate_shapes  # noqa: E402

FAMILIES = [  # first match wins, on the lowercase short name
    ("communication", r"allreduce|all_reduce|allgather|all_gather|reduce_scatter|nccl|cross_device|custom_ar|twoshot|oneshot"),
    ("expert_gemm", r"bmm_|fused_moe|moe_gemm|grouped_gemm|cutlass_moe"),
    ("kv_write", r"reshape_and_cache|kv_cache|cache_kernel"),
    # GEMMs over activations, not over a linear layer's weights: hyper-connection mixing, sparse-attention indexer scores
    ("activation_gemm", r"hc_prenorm|mhc|mqa_logits|indexer"),
    ("attention", r"fmha|flash_fwd|^fwd_kernel$|paged_attention|unified_attention|decode_attention|mla_|_mla|flashinfer.*(decode|prefill|attention)"),
    ("ssm_update", r"delta_rule|selective_state|ssd_|mamba_chunk|ssm"),
    ("conv", r"causal_conv1d"),
    ("gemm", r"nvjet|gemm|cutlass.*linear|linear.*cutlass|xmma|cublas|matmul|^sm\d+_.*(gemm|linear)"),
]
NOT_GEMM = r"splitkreduce|reduce_kernel|quantiz"


def kernel_models(ops: pd.DataFrame, steps: pd.DataFrame, info: dict, hw: dict) -> tuple[pd.DataFrame, dict]:
    """Adds model_sol_ns and model_family to the kernels of steady steps, returns the mapping checks."""
    ops = ops.assign(model_sol_ns=np.nan, model_memory_ns=np.nan, model_compute_ns=np.nan,
                     model_read_bytes=np.nan, model_write_bytes=np.nan, model_family=None,
                     model_quality="unsupported", model_reason="no semantic mapping", model_assumptions="[]", model_flops="{}")
    kernels = ops.kind == "kernel"
    ops.loc[kernels, "model_family"] = [family(row.name, row.signature) for row in ops[kernels].itertuples()]
    steady = steps[steps.steady]
    ordered = ops[kernels & ops.step.isin(steady.step)].sort_values(["step", "launch_start", "graphNodeId", "start"])
    checks: dict[str, str] = {}
    templates = {owner: gemm_template(ordered, steady, owner, info, hw, checks) for owner in OWNERS}
    by_step = dict(tuple(ordered.groupby("step")))
    for step in steady.itertuples():
        sols = step_models(by_step.get(step.step, ordered.iloc[:0]), step, info, hw, templates, checks)
        if sols:
            records = pd.DataFrame.from_dict({i: estimate_record(value) for i, value in sols.items()}, orient="index")
            ops.loc[records.index, records.columns] = records
    return ops, checks


def shape_models(ops: pd.DataFrame, info: dict, hw: dict, dtype: str | None = None) -> pd.DataFrame:
    """Native input shapes refine single-kernel operators. Precision must be declared, not guessed from shapes."""
    ops = ops.assign(operator_shape_estimate=False, operator_estimate_reason=None)
    for field, value in estimate_record(unsupported("", "no semantic mapping")).items():
        if field not in ops:
            ops[field] = value
    known = ops.operator_name.notna() & (ops.operator_kernel_count == 1)
    ops.loc[ops.operator_name.notna() & ~known, "operator_estimate_reason"] = "operator spans multiple GPU kernels"
    candidates = ops[known].assign(layer_mapped=ops.model_sol_ns[known].notna())
    keys = ["operator_name", "operator_input_shapes", "model_family", "layer_mapped"]
    for _, group in candidates.groupby(keys, dropna=False):
        row = next(group.itertuples())
        shapes = json.loads(row.operator_input_shapes)
        precision = dtype or mapped_precision(row, shapes, info)
        value = estimate_shapes(row.operator_name, shapes, precision, hardware_profile(info, hw))
        ops.loc[group.index, "operator_estimate_reason"] = value.reason
        if value.bound_ns is None or value.bound_ns <= 0:
            continue
        for field, entry in estimate_record(value).items():
            ops.loc[group.index, field] = entry
        ops.loc[group.index, "model_family"] = "nsys " + row.operator_name
        ops.loc[group.index, "operator_shape_estimate"] = True
    return ops


def mapped_precision(op, shapes: list[list[int]], info: dict) -> str | None:
    """Require an already-mapped GEMM plus consistent declared layer precision and matching logical dimensions."""
    if op.model_family != "gemm" or pd.isna(op.model_sol_ns) or op.operator_name != "aten::mm" or len(shapes) != 2:
        return None
    if len(shapes[1]) != 2:
        return None
    k = shapes[1][0]
    n = shapes[1][1]
    dtypes = {layer["compute_dtype"] for layers in info.get("layers", {}).values() for layer in layers
              if layer.get("compute_dtype") and layer.get("input_size") == k and layer.get("output_size") == n}
    return next(iter(dtypes)) if len(dtypes) == 1 else None


OWNERS = {"target": "execute_model", "draft": "propose"}  # the CPU phase that launches each forward


def family(name: str, signature: str = "") -> str:
    lowered = classification_name(name, signature).lower()
    if re.search(NOT_GEMM, lowered):
        return "elementwise"
    return next((label for label, pattern in FAMILIES if re.search(pattern, lowered)), "elementwise")


def classification_name(name: str, signature: str) -> str:
    """CUTLASS Kernel2 hides the operation in its template arguments, retained in the full symbol."""
    return signature.split("|", 1)[0] if name == "Kernel2" and signature else name


def step_models(ops: pd.DataFrame, step, info: dict, hw: dict, templates: dict, checks: dict) -> dict:
    """SOL in ns per kernel index of one step, kernels in program order."""
    model = Model(info, hw, step)
    sols = {i: model.communication() for i, f in ops.model_family.items() if f == "communication"}
    for owner, method in OWNERS.items():
        sols.update(Model(info, hw, step, owner).forward(ops[ops.phase.str.contains(method)], owner, templates[owner], checks))
    sols.update(model.sample(ops[ops.phase.str.contains(r"\.sample(?:$|>)", regex=True)]))
    return sols


def gemm_template(ordered: pd.DataFrame, steady: pd.DataFrame, owner: str, info: dict, hw: dict, checks: dict) -> dict:
    """Which linear layers each GEMM kernel of a forward computes, per GEMM count: steps with different token counts
    replay different CUDA graphs, whose GEMM counts can differ. Returns {count: [group of layers, or None] per kernel}.

    The kernels and the linear layers are both in program order, but rarely one to one: vLLM fuses some linears into
    one GEMM (a small gate next to a big projection), some linears launch no GEMM in decode, and some GEMMs belong to
    no linear layer. fit() searches the smallest such change that lines the two sequences up.
    """
    gemms = ordered[ordered.phase.str.contains(OWNERS[owner]) & (ordered.model_family == "gemm")]
    layers = [layer for layer in info["layers"][owner] if is_gemm_layer(layer, owner)]
    counts = gemms.groupby("step").size()
    templates = {}
    for count, steps in counts.groupby(counts).groups.items():
        template = fit(layers, gemms[gemms.step == steps[0]].name.tolist())
        mapped = sum(group is not None for group in template) if template else 0
        checks[f"{owner} GEMM mapping, {count} kernels ({len(steps)} of {steady.step.nunique()} steps)"] = f"{mapped} mapped" if template else "not mapped"
        if template:
            templates[count] = template
    return templates


def fit(layers: list[dict], names: list[str], candidates: int = 8) -> list | None:
    """Changes are whole classes: a layer class (the same name in every decoder layer, e.g. 'mlp.gate') is merged
    into an adjacent layer of its module or dropped, a rare kernel name is dropped. Only the `candidates` layer
    classes with the fewest bytes change (gates, aligners, indexers), plans with fewer changes and fewer bytes
    moved come first. A plan is kept only if every layer class then maps onto one or two kernel names over all
    layers, otherwise the order did not line up."""
    for merged, dropped, dropped_kernels in plans(layers, names, candidates):
        kept = [n for n in names if n not in dropped_kernels]
        groups = regroup(layers, merged, dropped)
        if groups is not None and len(groups) == len(kept) and is_consistent(groups, kept):
            return spread(groups, names, dropped_kernels)
    return None


def plans(layers: list[dict], names: list[str], candidates: int) -> list[tuple[set, set, set]]:
    """(merged classes, dropped classes, dropped kernel names) whose counts line up, best first."""
    counts = Counter(layer_class(layer["name"]) for layer in layers)
    nbytes: Counter = Counter()
    for layer in layers:
        nbytes[layer_class(layer["name"])] += layer["bytes"]
    cheap = sorted(counts, key=nbytes.get)[:candidates]
    kernels = Counter(names)
    rare = [n for n in kernels if kernels[n] < max(kernels.values())]
    found = []
    for actions in itertools.product(("keep", "merge", "drop"), repeat=len(cheap)):
        merged = {c for c, action in zip(cheap, actions) if action == "merge"}
        dropped = {c for c, action in zip(cheap, actions) if action == "drop"}
        groups = len(layers) - sum(counts[c] for c in merged | dropped)
        for r in range(len(rare) + 1):
            for dropped_kernels in itertools.combinations(rare, r):
                if groups == len(names) - sum(kernels[n] for n in dropped_kernels):
                    cost = (len(merged) + len(dropped) + r, sum(nbytes[c] for c in merged | dropped))
                    found.append((cost, merged, dropped, set(dropped_kernels)))
    return [plan[1:] for plan in sorted(found, key=lambda plan: plan[0])]


def regroup(layers: list[dict], merged: set[str], dropped: set[str]) -> list[list[dict]] | None:
    kept = [layer for layer in layers if layer_class(layer["name"]) not in dropped]
    return merge_classes(kept, merged) if merged else [[layer] for layer in kept]


def spread(groups: list[list[dict]], names: list[str], dropped: set[str]) -> list:
    """Groups onto the kernels in order, None for the dropped kernel names."""
    queue = iter(groups)
    return [None if name in dropped else next(queue) for name in names]


def is_gemm_layer(layer: dict, owner: str) -> bool:
    """Linear layers computed by a GEMM. The target's LM head runs in the sample phase, the draft's inside propose().
    Conv weights of recurrent layers are registered as linear layers but run in the conv kernel."""
    if layer["kind"] == "lm_head":
        return owner == "draft"
    return layer["kind"] == "linear" and not re.search(r"conv", layer["name"])


def merge_classes(layers: list[dict], merged: set[str]) -> list[list[dict]] | None:
    """Every layer of a merged class joins its previous neighbor of the same module, or else its next one."""
    groups: list[list[dict]] = []
    pending: list[dict] = []
    for layer in layers:
        joins = layer_class(layer["name"]) in merged
        if joins and groups and module(groups[-1][0]["name"]) == module(layer["name"]):
            groups[-1].append(layer)
        elif joins:
            pending.append(layer)
        elif pending and module(pending[0]["name"]) != module(layer["name"]):
            return None
        else:
            groups.append(pending + [layer])
            pending = []
    return None if pending else groups


def is_consistent(groups: list[list[dict]], names: list[str]) -> bool:
    by_class: dict[str, set[str]] = {}
    for group, name in zip(groups, names):
        by_class.setdefault(layer_class(group[0]["name"]), set()).add(name)
    return all(len(kernels) <= 2 for kernels in by_class.values())


def layer_class(name: str) -> str:
    """'model.layers.3.mlp.gate_up_proj' -> 'mlp.gate_up_proj'."""
    return re.sub(r".*?layers\.\d+\.", "", name)


def module(name: str) -> str:
    """'model.layers.3.mlp.shared_expert.down_proj' -> 'model.layers.3.mlp', the unit vLLM fuses within."""
    match = re.match(r"(.*?layers\.\d+\.[^.]+)", name)
    return match.group(1) if match else name.rsplit(".", 1)[0]


def group_sol(group: list[dict], model: "Model") -> Estimate:
    if len(group) == 1:
        return model.gemm(group[0])
    return model.gemm({"bytes": sum(layer["bytes"] for layer in group),
                       "elems": sum(layer.get("logical_elems", layer["elems"]) for layer in group)})


def estimate_record(value: Estimate) -> dict:
    return {"model_sol_ns": value.bound_ns, "model_memory_ns": value.memory_ns, "model_compute_ns": value.compute_ns,
            "model_read_bytes": value.work.read_bytes if value.work else None,
            "model_write_bytes": value.work.write_bytes if value.work else None,
            "model_quality": value.quality, "model_reason": value.reason,
            "model_assumptions": json.dumps(value.work.assumptions if value.work else ()),
            "model_flops": json.dumps(value.work.flops if value.work else {})}


class Model:
    """The bounds of one step's kernels, given the layer inventory and the step's tokens, requests and context."""

    def __init__(self, info: dict, hw: dict, step, owner: str = "target") -> None:
        self.info = info
        self.hw = hw
        self.hardware = hardware_profile(info, hw)
        self.tokens = query_tokens(step, owner, padded=True)
        logits_tokens = getattr(step, "logits_tokens", None)
        self.logits_tokens = float(logits_tokens) if pd.notna(logits_tokens) else None
        self.reqs = step.reqs
        self.context = step.context
        self.hidden = info.get("hidden_size", 0)
        self.world_size = hw.get("world_size", 1)
        self.shape_source = "draft execution shape unavailable" if owner == "draft" else str(getattr(step, "query_shape_source", "scheduler shape, execution shape unavailable"))
        self.queries = getattr(step, "query_lengths", None) if owner == "target" else None
        self.contexts = getattr(step, "context_lengths", None) if owner == "target" else None

    def traffic(self, read: float, write: float = 0, *assumptions: str) -> Estimate:
        return estimate("traffic", {"read_bytes": read, "write_bytes": write,
                                    "assumptions": (self.shape_source, *assumptions)}, self.hardware)

    def communication(self) -> Estimate:
        n = self.world_size
        bandwidth = self.hw.get("link_bytes_per_s")
        if n <= 1 or not bandwidth or self.tokens is None or not self.hidden:
            return unsupported("communication", "link bandwidth, execution shape or activation dimensions unavailable")
        traffic = 2 * (n - 1) / n * self.tokens * self.hidden * self.info.get("activation_bytes", 2)
        return estimate("traffic", {"read_bytes": traffic, "assumptions": ("ring all-reduce over declared link bandwidth",)}, Hardware(bandwidth))

    def forward(self, ops: pd.DataFrame, owner: str, templates: dict, checks: dict) -> dict:
        cache = [c for c in self.info["cache"] if c["owner"] == owner]
        moes = [layer for layer in self.info["layers"][owner] if layer["kind"] == "moe"]
        attention = [c for c in cache if c["kind"] == "attention"]
        states = [c for c in cache if c["kind"] == "state"]
        plan = [
            ("expert_gemm", [(m, part) for m in moes for part in ["w13", "w2"]], self.expert_gemm),
            ("attention", attention, self.attention),
            ("kv_write", attention, lambda c: self.traffic(0, c["kv_bytes_per_token"] * self.tokens, "new KV stored once") if self.tokens is not None else unsupported("kv_write", "execution token count unavailable")),
            ("ssm_update", states, lambda c: estimate("stateful", {"state_bytes": c["ssm_bytes"], "requests": self.reqs,
                                                                   "checkpoint_write_bytes": c["ssm_bytes"] * self.reqs}, self.hardware)),
            ("conv", states, lambda c: self.traffic(c["conv_bytes"] * self.reqs, 0, "state-read reference only, convolution work and persistence unspecified")),
            ("activation_gemm", [None], lambda _: self.activation_gemm()),
        ]
        sols = self.gemms(ops[ops.model_family == "gemm"], templates)
        for name, layers, cost in plan:
            kernels = ops[ops.model_family == name]
            mapped = assign(kernels, layers, cost)
            if mapped is None:
                checks[f"{owner} {name}"] = f"{len(kernels)} kernels for {len(layers)} layers, not mapped"
                mapped = {i: unsupported(name, "kernel-to-layer mapping unavailable") for i in kernels.index}
            sols.update(mapped)
        return sols

    def gemms(self, kernels: pd.DataFrame, templates: dict) -> dict:
        """GEMM kernels follow the template of their count, steps with an unmapped count get no model."""
        template = templates.get(len(kernels))
        if template is None:
            return {i: unsupported("linear", "kernel-to-layer mapping unavailable") for i in kernels.index}
        return {i: group_sol(group, self) if group else unsupported("linear", "kernel has no mapped layer") for i, group in zip(kernels.index, template)}

    def gemm(self, layer: dict) -> Estimate:
        meta = {"weight_bytes": layer["bytes"], "weight_elements": layer.get("logical_elems", layer["elems"]),
                "tokens": self.tokens, "compute_dtype": layer.get("compute_dtype", self.info.get("activation_dtype", "bf16")),
                "input_width": layer.get("input_size"), "output_width": layer.get("output_size"),
                "activation_bytes": self.info.get("activation_bytes", 2), "assumptions": (self.shape_source, "kernel-to-layer association inferred from execution order")}
        return estimate("linear", meta, self.hardware)

    def expert_gemm(self, moe_part: tuple[dict, str]) -> Estimate:
        moe, part = moe_part
        k = moe.get("top_k") or self.info["top_k"] or 1
        experts = moe.get("global_experts") or moe["experts"]  # the local experts are a uniform share of all of them
        touched = 1 - (1 - k / experts) ** self.reqs  # the tokens of one request (verify + drafts) route alike
        return estimate("linear", {"weight_bytes": moe[f"{part}_bytes"] * touched,
                                  "weight_elements": moe[f"{part}_elems"] * k / experts,
                                  "tokens": self.tokens, "compute_dtype": self.info.get("activation_dtype", "bf16"),
                                  "assumptions": (self.shape_source, "uniform expert routing, drafts of one request share experts")}, self.hardware)

    def activation_gemm(self) -> Estimate:
        """GEMMs over activations, not a linear layer's weights (hyper-connection mixing, sparse-attention indexer
        scores): no declared weight to size the estimate from, so the bound is one read + one write of the
        activation tensor, the floor a GEMM over it cannot beat."""
        if self.tokens is None or not self.hidden:
            return unsupported("activation_gemm", "execution shape or activation dimensions unavailable")
        size = self.tokens * self.hidden * self.info.get("activation_bytes", 2)
        return self.traffic(size, size, "single read + write of the activation tensor, GEMM operands are activations not a declared weight layer")

    def attention(self, cache: dict) -> Estimate:
        if cache.get("layout") == "FullAttentionSpec" and all(cache.get(key) for key in ("query_heads", "kv_heads", "head_dim", "kv_element_bytes")) and self.queries is not None and self.contexts is not None:
            return estimate("attention", {"query_lengths": self.queries, "context_lengths": self.contexts,
                                          "query_heads": cache["query_heads"], "kv_heads": cache["kv_heads"], "head_dim": cache["head_dim"],
                                          "activation_bytes": self.info.get("activation_bytes", 2), "kv_bytes": cache["kv_element_bytes"],
                                          "compute_dtype": self.info.get("activation_dtype", "bf16")}, self.hardware)
        # Older probes know cache bytes but not query heads, so retain only the declared KV traffic bound.
        tokens = self.tokens or 0
        return self.traffic(cache["kv_bytes_per_token"] * (self.context + tokens), 0,
                            "all cached KV read once, attention arithmetic and activation traffic unavailable")

    def sample(self, ops: pd.DataFrame) -> dict:
        """The target's logits: the LM head GEMM reads its weights, the sampling kernels read the logits."""
        heads = [layer for layer in self.info["layers"]["target"] if layer["kind"] == "lm_head"]
        # Sampling rows are not the padded target-forward rows. Without a logits shape use a weight-read bound.
        self.tokens = self.logits_tokens
        head = self.gemm(heads[0]) if heads else unsupported("linear", "LM head unavailable")
        return {i: head for i, f in ops.model_family.items() if f == "gemm"}


def hardware_profile(info: dict, hw: dict) -> Hardware:
    rates = dict(info.get("compute_flops_per_s", {}))
    if info.get("gemm_flops_per_s"):
        rates.setdefault(info.get("gemm_dtype", "bf16"), info["gemm_flops_per_s"])
    return Hardware(hw["bw"], rates)


def query_tokens(step, owner: str = "target", padded: bool = False) -> float | None:
    if owner == "draft":
        value = getattr(step, "draft_tokens", None)
        return float(value) if pd.notna(value) else None
    actual = getattr(step, "actual_tokens", None)
    padded_value = getattr(step, "padded_tokens", None)
    if padded and pd.notna(padded_value):
        return float(padded_value)
    if pd.notna(actual):
        return float(actual)
    scheduled = getattr(step, "tokens", None)
    return float(scheduled) if not getattr(step, "av_enabled", False) and pd.notna(scheduled) else None


def assign(kernels: pd.DataFrame, layers: list, cost) -> dict | None:
    """Kernels in program order onto the layers, which repeat when a step runs several forwards (count = m x layers).
    None when the count does not fit."""
    if kernels.empty:
        return {}
    if not layers or len(kernels) % len(layers):
        return None
    return {i: cost(layers[n % len(layers)]) for n, i in enumerate(kernels.index)}
