"""Semantic work and roofline bounds, independent of model classes and kernel names.

Memory bounds assume the declared traffic is served by the declared memory tier. They exclude launch latency,
cache effects and intermediate materialization. A missing compute ceiling leaves an explicitly partial bound.
"""

from dataclasses import dataclass, field
import math
from typing import Callable, Mapping


@dataclass(frozen=True)
class Hardware:
    memory_bytes_per_s: float
    flops_per_s: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Work:
    read_bytes: float
    write_bytes: float = 0
    flops: Mapping[str, float] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    complete: bool = True


@dataclass(frozen=True)
class Estimate:
    operation: str
    work: Work | None
    memory_ns: float | None
    compute_ns: float | None
    bound_ns: float | None
    quality: str
    reason: str = ""


def estimate(operation: str, metadata: dict, hardware: Hardware) -> Estimate:
    """Unknown operations or insufficient metadata remain unsupported, with no fabricated fallback."""
    model = OPERATIONS.get(operation)
    if model is None:
        return unsupported(operation, "no semantic work model")
    try:
        work = model(metadata)
    except (KeyError, TypeError, ValueError) as error:
        return unsupported(operation, f"insufficient or invalid metadata: {error}")
    return price(operation, work, hardware)


def price(operation: str, work: Work, hardware: Hardware) -> Estimate:
    values = [work.read_bytes, work.write_bytes, *work.flops.values()]
    if any(not math.isfinite(value) or value < 0 for value in values):
        return unsupported(operation, "work must be finite and nonnegative")
    if not math.isfinite(hardware.memory_bytes_per_s) or hardware.memory_bytes_per_s <= 0:
        return unsupported(operation, "missing memory bandwidth")
    memory = (work.read_bytes + work.write_bytes) / hardware.memory_bytes_per_s * 1e9
    missing = [dtype for dtype, count in work.flops.items() if count > 0 and not valid_rate(hardware.flops_per_s.get(dtype))]
    compute = None if missing else max((count / hardware.flops_per_s[dtype] * 1e9
                                       for dtype, count in work.flops.items() if count > 0), default=0)
    quality = "partial" if missing or not work.complete else "assumed" if work.assumptions else "specified"
    reason = "missing compute ceiling: " + ", ".join(missing) if missing else ""
    return Estimate(operation, work, memory, compute, max(memory, compute or 0), quality, reason)


def unsupported(operation: str, reason: str) -> Estimate:
    return Estimate(operation, None, None, None, None, "unsupported", reason)


def utilization_reference(duration_ns: float, compute_pct: float, memory_pct: float) -> Estimate:
    """Profiler resource-demand reference for operations lacking semantic metadata, independent of kernel names.

    This estimates time at peak utilization for the measured work, not an implementation-independent work bound.
    """
    if not math.isfinite(duration_ns) or duration_ns <= 0:
        return unsupported("counter_reference", "missing profiler duration")
    memory = duration_ns * memory_pct / 100 if math.isfinite(memory_pct) and memory_pct > 0 else None
    compute = duration_ns * compute_pct / 100 if math.isfinite(compute_pct) and compute_pct > 0 else None
    if memory is None and compute is None:
        return unsupported("counter_reference", "missing resource utilization")
    return Estimate("counter_reference", None, memory, compute, max(memory or 0, compute or 0),
                    "counter-reference", "resource-demand reference for measured work, excludes latency")


# Work models consume logical dimensions and byte sizes, never profiler kernel names.
def linear(meta: dict) -> Work:
    weight_bytes = number(meta, "weight_bytes")
    elements = number(meta, "weight_elements")
    tokens = meta.get("tokens")
    assumptions = tuple(meta.get("assumptions", ()))
    if tokens is None:
        return Work(weight_bytes, assumptions=assumptions + ("query shape unavailable, weight-read bound only",), complete=False)
    tokens = number(meta, "tokens")
    dtype = meta["compute_dtype"]
    widths = meta.get("input_width"), meta.get("output_width")
    if None in widths:
        return Work(weight_bytes, flops={dtype: 2 * elements * tokens}, complete=False,
                    assumptions=assumptions + ("activation dimensions unavailable, activation traffic excluded",))
    element_bytes = number(meta, "activation_bytes")
    return Work(weight_bytes + tokens * number(meta, "input_width") * element_bytes,
                tokens * number(meta, "output_width") * element_bytes,
                {dtype: 2 * elements * tokens}, assumptions)


def matmul(meta: dict) -> Work:
    m = number(meta, "m")
    n = number(meta, "n")
    k = number(meta, "k")
    batches = number(meta, "batches", 1)
    reads = batches * (m * k * number(meta, "a_bytes") + k * n * number(meta, "b_bytes"))
    return Work(reads, batches * m * n * number(meta, "output_bytes"),
                {meta["compute_dtype"]: 2 * batches * m * n * k})


def attention(meta: dict) -> Work:
    """Dense causal attention over ragged queries and an existing KV prefix, with GQA-aware KV traffic."""
    queries = tuple(meta["query_lengths"])
    contexts = tuple(meta["context_lengths"])
    if len(queries) != len(contexts) or any(q < 0 or c < 0 for q, c in zip(queries, contexts)):
        raise ValueError("query and context lengths must be matching nonnegative sequences")
    tokens = sum(queries)
    heads = number(meta, "query_heads")
    kv_heads = number(meta, "kv_heads")
    dim = number(meta, "head_dim")
    pairs = sum(q * c + q * (q + 1) / 2 for q, c in zip(queries, contexts))
    element_bytes = number(meta, "activation_bytes")
    reads = tokens * heads * dim * element_bytes + sum(q + c for q, c in zip(queries, contexts)) * 2 * kv_heads * dim * number(meta, "kv_bytes")
    writes = tokens * heads * dim * element_bytes
    return Work(reads, writes, {meta["compute_dtype"]: 4 * heads * dim * pairs},
                ("dense causal attention, KV read once per request", "softmax nonlinear work excluded"), complete=False)


def convolution(meta: dict) -> Work:
    tokens = number(meta, "tokens")
    inputs = number(meta, "input_channels")
    outputs = number(meta, "output_channels")
    width = number(meta, "kernel_width")
    groups = number(meta, "groups", 1)
    element_bytes = number(meta, "activation_bytes")
    weight_bytes = number(meta, "weight_bytes")
    return Work(tokens * inputs * element_bytes + weight_bytes, tokens * outputs * element_bytes,
                {meta["compute_dtype"]: 2 * tokens * outputs * inputs / groups * width},
                ("each input and weight read once, output written once",))


def pointwise(meta: dict) -> Work:
    elements = number(meta, "elements")
    return Work(number(meta, "read_bytes"), number(meta, "write_bytes"),
                {meta["compute_dtype"]: elements * number(meta, "flops_per_element")})


def reduction(meta: dict) -> Work:
    elements = number(meta, "elements")
    outputs = number(meta, "output_elements")
    if outputs > elements:
        raise ValueError("reduction outputs exceed inputs")
    return Work(elements * number(meta, "input_bytes"), outputs * number(meta, "output_bytes"),
                {meta["compute_dtype"]: elements - outputs})


def stateful(meta: dict) -> Work:
    """One initial state read. Persistence is a declared amortized budget, never a per-query snapshot rule."""
    state = number(meta, "state_bytes") * number(meta, "requests")
    writes = number(meta, "checkpoint_write_bytes", 0)
    flops = meta.get("flops", {})
    return Work(state + number(meta, "input_bytes", 0), writes + number(meta, "output_bytes", 0), flops,
                ("initial state read once", "checkpoint writes use the declared amortized budget",
                 "no intermediate snapshots or layout copies"), complete=bool(flops) and "input_bytes" in meta and "output_bytes" in meta)


def traffic(meta: dict) -> Work:
    return Work(number(meta, "read_bytes"), number(meta, "write_bytes", 0),
                assumptions=tuple(meta.get("assumptions", ())), complete=False)


def copy_work(meta: dict) -> Work:
    payload = number(meta, "bytes")
    kind = meta["kind"]
    if kind not in ("device_to_device", "memset"):
        raise ValueError("copy memory tier unavailable or not device memory")
    return Work(payload if kind == "device_to_device" else 0, payload,
                assumptions=("device memory traffic, cache and copy-engine latency excluded",))


OPERATIONS: dict[str, Callable[[dict], Work]] = {
    "linear": linear, "matmul": matmul, "attention": attention, "convolution": convolution,
    "pointwise": pointwise, "reduction": reduction, "stateful": stateful, "traffic": traffic, "copy": copy_work,
}


def number(meta: dict, key: str, default: float | None = None) -> float:
    value = float(meta[key] if default is None else meta.get(key, default))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{key} must be finite and nonnegative")
    if key == "groups" and value == 0:
        raise ValueError("groups must be positive")
    return value


def valid_rate(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0
