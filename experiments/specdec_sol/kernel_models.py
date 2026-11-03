"""Analytic SOL per kernel, from the layer inventory the probe reports and the step's batch shape.

Kernels of one forward run in program order, so the k-th GEMM kernel belongs to the k-th linear layer, the k-th
attention kernel to the k-th attention layer, and so on. A family is mapped only when its kernel count is a multiple
of its layer count (several forwards per step, or several kernels per layer), otherwise its kernels get no model and
the mismatch is reported. Each kernel's bound is max(bytes / peak DRAM BW, FLOPs / measured GEMM rate):

- gemm: the layer's weights, 2 x elements x tokens FLOPs.
- expert_gemm: per MoE layer w13 then w2, only the experts touched with one routing per request (uniform over requests).
- attention: the layer's KV of all context tokens. kv_write: the KV of the new tokens.
- ssm_update / conv: read and write the recurrent SSM / conv state of every request.
- lm_head and sampling (sample phase): the LM head weights, the logits of the step's tokens.
- communication: a ring all-reduce of one activation vector per token, 2 (n - 1) / n of it per GPU over NVLink.
- activation_gemm: GEMMs whose operands are activations, not a layer's weights (e.g. hyper-connection mixing):
  read and write one activation vector per token, like elementwise kernels.
- elementwise (everything else): read and write one activation vector per token, what a fused kernel cannot avoid.
"""

import itertools
import re
from collections import Counter

import numpy as np
import pandas as pd

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
ACT_BYTES = 2  # bf16 activations
NVLINK_BYTES_PER_S = 900e9  # NVLink 5 (B200/B300) per direction and GPU, nsys does not record it


def kernel_models(ops: pd.DataFrame, steps: pd.DataFrame, info: dict, hw: dict) -> tuple[pd.DataFrame, dict]:
    """Adds model_sol_ns and model_family to the kernels of steady steps, returns the mapping checks."""
    ops = ops.assign(model_sol_ns=np.nan, model_family=None)
    kernels = ops.kind == "kernel"
    ops.loc[kernels, "model_family"] = [family(row.name, row.signature) for row in ops[kernels].itertuples()]
    steady = steps[steps.steady]
    ordered = ops[kernels & ops.step.isin(steady.step)].sort_values(["step", "launch_start", "graphNodeId", "start"])
    checks: dict[str, str] = {}
    templates = {owner: gemm_template(ordered, steady, owner, info, hw, checks) for owner in OWNERS}
    by_step = dict(tuple(ordered.groupby("step")))
    for step in steady.itertuples():
        sols = step_models(by_step.get(step.step, ordered.iloc[:0]), step, info, hw, templates, checks)
        ops.loc[list(sols), "model_sol_ns"] = list(sols.values())
    return ops, checks


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
    sols = {i: model.communication() if f == "communication" else model.elementwise() for i, f in ops.model_family.items()}
    for owner, method in OWNERS.items():
        sols.update(model.forward(ops[ops.phase.str.contains(method)], owner, templates[owner], checks))
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
        checks[f"{owner} GEMM mapping, {count} kernels ({len(steps)} steps)"] = f"{mapped} mapped" if template else "not mapped"
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


def group_sol(group: list[dict], model: "Model") -> float:
    return model.gemm({"bytes": sum(layer["bytes"] for layer in group), "elems": sum(layer["elems"] for layer in group)})


class Model:
    """The bounds of one step's kernels, given the layer inventory and the step's tokens, requests and context."""

    def __init__(self, info: dict, hw: dict, step) -> None:
        self.info = info
        self.bw = hw["bw"]
        self.flops = info["gemm_flops_per_s"]
        self.tokens = step.tokens
        self.reqs = step.reqs
        self.context = step.context
        self.hidden = info["hidden_size"]
        self.world_size = hw.get("world_size", 1)

    def bound(self, nbytes: float, flops: float = 0.0) -> float:
        return max(nbytes / self.bw, flops / self.flops) * 1e9

    def communication(self) -> float:
        n = self.world_size
        return 2 * (n - 1) / n * self.tokens * self.hidden * ACT_BYTES / NVLINK_BYTES_PER_S * 1e9 if n > 1 else np.nan

    def elementwise(self) -> float:
        return self.bound(2 * self.tokens * self.hidden * ACT_BYTES)

    def forward(self, ops: pd.DataFrame, owner: str, templates: dict, checks: dict) -> dict:
        cache = [c for c in self.info["cache"] if c["owner"] == owner]
        moes = [layer for layer in self.info["layers"][owner] if layer["kind"] == "moe"]
        attention = [c for c in cache if c["kind"] == "attention"]
        states = [c for c in cache if c["kind"] == "state"]
        plan = [
            ("expert_gemm", [(m, part) for m in moes for part in ["w13", "w2"]], self.expert_gemm),
            ("attention", attention, lambda c: self.bound(c["kv_bytes_per_token"] * (self.context + self.tokens))),
            ("kv_write", attention, lambda c: self.bound(c["kv_bytes_per_token"] * self.tokens)),
            ("ssm_update", states, lambda c: self.bound(2 * c["ssm_bytes"] * self.reqs)),
            ("conv", states, lambda c: self.bound(2 * c["conv_bytes"] * self.reqs)),
        ]
        sols = self.gemms(ops[ops.model_family == "gemm"], templates)
        for name, layers, cost in plan:
            kernels = ops[ops.model_family == name]
            mapped = assign(kernels, layers, cost)
            if mapped is None:
                checks[f"{owner} {name}"] = f"{len(kernels)} kernels for {len(layers)} layers, not mapped"
                mapped = {i: np.nan for i in kernels.index}
            sols.update(mapped)
        return sols

    def gemms(self, kernels: pd.DataFrame, templates: dict) -> dict:
        """GEMM kernels follow the template of their count, steps with an unmapped count get no model."""
        template = templates.get(len(kernels))
        if template is None:
            return {i: np.nan for i in kernels.index}
        return {i: group_sol(group, self) if group else np.nan for i, group in zip(kernels.index, template)}

    def gemm(self, layer: dict) -> float:
        return self.bound(layer["bytes"], 2.0 * layer["elems"] * self.tokens)

    def expert_gemm(self, moe_part: tuple[dict, str]) -> float:
        moe, part = moe_part
        k = moe.get("top_k") or self.info["top_k"] or 1
        experts = moe.get("global_experts") or moe["experts"]  # the local experts are a uniform share of all of them
        touched = 1 - (1 - k / experts) ** self.reqs  # the tokens of one request (verify + drafts) route alike
        flops = 2 * moe[f"{part}_elems"] / moe["experts"] * k * moe["experts"] / experts * self.tokens
        return self.bound(moe[f"{part}_bytes"] * touched, flops)

    def sample(self, ops: pd.DataFrame) -> dict:
        """The target's logits: the LM head GEMM reads its weights, the sampling kernels read the logits."""
        heads = [layer for layer in self.info["layers"]["target"] if layer["kind"] == "lm_head"]
        vocab = heads[0]["elems"] / self.hidden if heads and self.hidden else 0
        logits = self.bound(self.tokens * vocab * 4)
        head = self.gemm(heads[0]) if heads else np.nan
        return {i: head if f == "gemm" else logits for i, f in ops.model_family.items()}


def assign(kernels: pd.DataFrame, layers: list, cost) -> dict | None:
    """Kernels in program order onto the layers, which repeat when a step runs several forwards (count = m x layers).
    None when the count does not fit."""
    if kernels.empty:
        return {}
    if not layers or len(kernels) % len(layers):
        return None
    return {i: cost(layers[n % len(layers)]) for n, i in enumerate(kernels.index)}
