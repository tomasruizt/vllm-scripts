"""Triton core-path top-k/top-p filter (research prototype, not for vLLM).

1. Levels (rows x splits): stream each row once, write the max of every group of 16
   logits (level 1) and of every 256 (level 2), and the row's max and sum of exp for
   top-p-only rows. No compaction, so it runs close to the streaming bandwidth.
2. Select (one program per row): select exactly through the levels. The supergroups
   at or above the n-th best supergroup max hold the best n groups, and the groups at
   or above the n-th best group max hold every logit at or above it. Tiers of C = 32,
   64 (top-k only) and 128 logits, with tl.topk at each level. Writes the kept keys.
3. Wide (top-p-only rows the tiers missed): a threshold t from 8 supergroup-max ranks
   whose group maxima alone reach the nucleus mass, then compaction of the level-1
   groups and logits at or above t (at most 4096) and a sort.
4. Fill (rows x splits): -inf over the row, then scatter the kept logits.
Rows that need more (k > 128, a larger nucleus, too many ties at a cut) stay untouched
and the existing Triton kernel masks them, with the other rows' k and p set to no-ops.
Same semantics as the CUDA filter: exactly k, ties by the lowest index.
"""

import functools

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

BLOCK = 4096  # elements per streaming step, a multiple of 256
MAX_SPLITS = 32
TIERS = (32, 64, 128)  # logits per tier (64 for top-k only)
WIDE = 4096  # candidate logits of the wide tier
WIDE_GROUPS = 4096  # candidate groups of the wide tier
NOOP_C = tl.constexpr(-1)  # row states besides the number of kept logits
WIDE_PENDING_C = tl.constexpr(-2)
LAUNCHED: dict = {}  # compiled kernels of the last call, for inspection (v2: 4+ launches, kept for comparison)


def apply_top_k_top_p_triton_core(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    has_topp_only_rows: bool = True,
) -> torch.Tensor:
    """Mask logits in place outside the per-row top-k, then the top-p nucleus."""
    if k is None and p is None:
        return logits
    rows, vocab = logits.shape
    splits = num_splits(rows, logits.device)
    chunk = triton.cdiv(triton.cdiv(vocab, splits), BLOCK) * BLOCK
    groups, supergroups = triton.cdiv(vocab, 16), triton.cdiv(vocab, 256)
    level1, level2, stats, state, fallback_k, fallback_p, scratch = buffers(
        logits.device, rows, splits, vocab
    )
    k_arg = k if k is not None else fallback_k
    p_arg = p if p is not None else fallback_p
    shape = dict(vocab=vocab, groups=groups, supergroups=supergroups, SPLITS=splits)
    params = dict(k_ptr=k_arg, p_ptr=p_arg, HAS_K=k is not None, HAS_P=p is not None)
    LAUNCHED["levels"] = _levels_kernel[(rows, splits)](
        logits, logits.stride(0), level1, level2, stats, chunk, **shape, **params, BLOCK=BLOCK,
        num_warps=4,
    )  # fmt: skip
    rows_args = (logits, logits.stride(0), level1, level2, stats, state, scratch)
    p2 = triton.next_power_of_2(supergroups)
    LAUNCHED["select"] = _select_kernel[(rows,)](
        *rows_args, fallback_k, fallback_p, **shape, **params, SUPERGROUPS_P2=p2,
        SMALL=TIERS[0], MID=TIERS[1], LARGE=TIERS[2], WIDE=WIDE, WIDE_GROUPS=WIDE_GROUPS,
        num_warps=8,
    )  # fmt: skip
    if p is not None and has_topp_only_rows:
        LAUNCHED["wide"] = _wide_kernel[(rows,)](
            *rows_args, fallback_p, **shape, SUPERGROUPS_P2=p2, WIDE=WIDE,
            WIDE_GROUPS=WIDE_GROUPS, BLOCK=BLOCK, num_warps=8,
        )  # fmt: skip
    LAUNCHED["fill"] = _fill_kernel[(rows, splits)](
        logits, logits.stride(0), state, scratch, vocab, chunk, WIDE=WIDE,
        WIDE_GROUPS=WIDE_GROUPS, BLOCK=BLOCK, num_warps=4,
    )  # fmt: skip
    return apply_top_k_top_p_triton(
        logits,
        fallback_k if k is not None else None,
        fallback_p if p is not None else None,
        has_topp_only_rows=has_topp_only_rows,
    )


def num_splits(rows: int, device: torch.device) -> int:
    """Split rows for streaming until the grid has about 4 programs per SM."""
    return min(MAX_SPLITS, triton.next_power_of_2(triton.cdiv(4 * num_sms(device), rows)))


@functools.cache
def num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


_BUFFERS: dict[tuple, tuple[torch.Tensor, ...]] = {}


def buffers(device: torch.device, rows: int, splits: int, vocab: int):
    """Level 1 and 2 maxima, (max, sum of exp) per split, row states, fallback k and
    p, and per row the wide tier's candidate groups and the kept (candidate) keys."""
    key = (device, rows, splits, vocab)
    if key not in _BUFFERS:
        _BUFFERS[key] = (
            torch.empty(rows * triton.cdiv(vocab, 16), dtype=torch.float32, device=device),
            torch.empty(rows * triton.cdiv(vocab, 256), dtype=torch.float32, device=device),
            torch.empty(rows * splits * 2, dtype=torch.float32, device=device),
            torch.empty(rows, dtype=torch.int32, device=device),
            torch.empty(rows, dtype=torch.int32, device=device),
            torch.empty(rows, dtype=torch.float32, device=device),
            torch.empty(rows * (WIDE_GROUPS // 2 + WIDE), dtype=torch.int64, device=device),
        )
    return _BUFFERS[key]


@triton.jit
def _levels_kernel(
    logits_ptr, stride, level1_ptr, level2_ptr, stats_ptr, chunk, vocab, groups,
    supergroups, k_ptr, p_ptr, HAS_K: tl.constexpr, HAS_P: tl.constexpr,
    SPLITS: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    row = tl.program_id(0)
    split = tl.program_id(1)
    k, p = _row_params(k_ptr, p_ptr, row, vocab, HAS_K, HAS_P)
    if (k >= vocab) & (p >= 1.0):
        return
    row_ptr = logits_ptr + row.to(tl.int64) * stride
    level1_row = level1_ptr + row.to(tl.int64) * groups
    level2_row = level2_ptr + row.to(tl.int64) * supergroups
    with_stats = (p < 1.0) & (k >= vocab)
    offs = tl.arange(0, BLOCK)
    lo = split * chunk
    hi = tl.minimum(lo + chunk, vocab)
    row_max = -3.0e38
    row_sum = 0.0
    for start in range(lo, hi, BLOCK):
        idx = start + offs
        x = tl.load(row_ptr + idx, mask=idx < vocab, other=-float("inf"))
        group_max = tl.max(tl.reshape(x, [BLOCK // 16, 16]), 1)
        group = start // 16 + tl.arange(0, BLOCK // 16)
        tl.store(level1_row + group, group_max, mask=group < groups)
        supergroup = start // 256 + tl.arange(0, BLOCK // 256)
        supergroup_max = tl.max(tl.reshape(group_max, [BLOCK // 256, 16]), 1)
        tl.store(level2_row + supergroup, supergroup_max, mask=supergroup < supergroups)
        if with_stats:
            new_max = tl.maximum(row_max, tl.max(x))
            row_sum = row_sum * tl.exp(row_max - new_max) + tl.sum(tl.exp(x - new_max))
            row_max = new_max
    program = row * SPLITS + split
    tl.store(stats_ptr + 2 * program, row_max)
    tl.store(stats_ptr + 2 * program + 1, row_sum)


@triton.jit
def _select_kernel(
    logits_ptr, stride, level1_ptr, level2_ptr, stats_ptr, state_ptr, scratch_ptr,
    fallback_k_ptr, fallback_p_ptr, vocab, groups, supergroups, k_ptr, p_ptr,
    HAS_K: tl.constexpr, HAS_P: tl.constexpr, SPLITS: tl.constexpr,
    SUPERGROUPS_P2: tl.constexpr, SMALL: tl.constexpr, MID: tl.constexpr,
    LARGE: tl.constexpr, WIDE: tl.constexpr, WIDE_GROUPS: tl.constexpr,
):  # fmt: skip
    row = tl.program_id(0)
    k, p = _row_params(k_ptr, p_ptr, row, vocab, HAS_K, HAS_P)
    no_k = k >= vocab
    done = no_k & (p >= 1.0)
    kept = -1
    if not done:
        row_ptr = logits_ptr + row.to(tl.int64) * stride
        kept_row = scratch_ptr + row.to(tl.int64) * (WIDE_GROUPS // 2 + WIDE) + WIDE_GROUPS // 2
        row_max, row_sum = _row_stats(stats_ptr, row, SPLITS)
        s = tl.arange(0, SUPERGROUPS_P2)
        level2 = tl.load(level2_ptr + row.to(tl.int64) * supergroups + s, mask=s < supergroups,
                         other=-float("inf"))  # fmt: skip
        args = (row_ptr, level1_ptr + row.to(tl.int64) * groups, kept_row, _key(level2, s), level2,
                k, p, row_max, row_sum, vocab, groups)  # fmt: skip
        if no_k | (k <= SMALL):
            done, kept = _tier(*args, SMALL)
        if (not done) & (k <= MID):
            done, kept = _tier(*args, MID)
        if (not done) & (no_k | (k <= LARGE)):
            done, kept = _tier(*args, LARGE)
    state = tl.where(done, kept, tl.where(no_k, WIDE_PENDING_C, NOOP_C))
    tl.store(state_ptr + row, state)
    tl.store(fallback_k_ptr + row, tl.where(done, vocab, k))
    tl.store(fallback_p_ptr + row, tl.where(done, 1.0, p))


@triton.jit
def _tier(row_ptr, level1_row, kept_row, keys2, level2, k, p, row_max, row_sum, vocab, groups,
          C: tl.constexpr):  # fmt: skip
    """Write the kept keys if the row's best C logits decide the cutoff, return
    whether they did and how many are kept."""
    use_k = k < vocab
    use_p = p < 1.0
    rank = tl.arange(0, C)
    # Top-p only needs C / 2, leaving room for ties at the cuts.
    need = tl.where(use_k, k, C // 2)
    ok = need <= C
    # The supergroups at or above the need-th best max hold the best need groups.
    top2 = tl.topk(keys2, C, descending=False)
    cut2 = tl.sum(tl.where(rank == need - 1, _key_value(top2), 0.0))
    count2 = tl.sum((level2 >= cut2).to(tl.int32))
    ok = ok & (count2 <= C)
    group = (top2 & 0xFFFFFFFF)[:, None] * 16 + tl.arange(0, 16)[None, :]
    level1 = tl.load(level1_row + group, mask=(rank < count2)[:, None] & (group < groups),
                     other=-float("inf"))  # fmt: skip
    # The groups at or above the need-th best group max hold every logit at or above
    # it (at least need logits).
    top1 = tl.topk(tl.reshape(_key(level1, group), [C * 16]), C, descending=False)
    cut1 = tl.sum(tl.where(rank == need - 1, _key_value(top1), 0.0))
    count1 = tl.sum((tl.reshape(level1, [C * 16]) >= cut1).to(tl.int32))
    ok = ok & (count1 <= C)
    index = (top1 & 0xFFFFFFFF)[:, None] * 16 + tl.arange(0, 16)[None, :]
    x = tl.load(row_ptr + index, mask=(rank < count1)[:, None] & (index < vocab),
                other=-float("inf"))  # fmt: skip
    # The best logits in exact order (logit descending, index ascending), exact up to
    # the last one at or above cut1.
    top = tl.topk(tl.reshape(_key(x, index), [C * 16]), C, descending=False)
    value = _key_value(top)
    keep = (rank < k) & (value >= cut1)
    if use_p:
        best = tl.max(value)
        weight = tl.where(keep, tl.exp(value - best), 0.0)
        kept_mass = tl.sum(weight)
        total = tl.where(use_k, kept_mass, row_sum * tl.exp(row_max - best))
        keep = keep & (tl.cumsum(weight, 0) - weight < p * total)
        ok = ok & (use_k | (kept_mass >= p * total))
    tl.store(kept_row + rank, top, mask=keep & ok)
    return ok, tl.sum(keep.to(tl.int32))


@triton.jit
def _wide_kernel(
    logits_ptr, stride, level1_ptr, level2_ptr, stats_ptr, state_ptr, scratch_ptr,
    fallback_p_ptr, vocab, groups, supergroups, SPLITS: tl.constexpr,
    SUPERGROUPS_P2: tl.constexpr, WIDE: tl.constexpr, WIDE_GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
):  # fmt: skip
    """Top-p only: mask the row from the logits at or above t, if at most WIDE_GROUPS
    groups and WIDE logits reach t."""
    row = tl.program_id(0)
    if tl.load(state_ptr + row) != WIDE_PENDING_C:
        return
    row_ptr = logits_ptr + row.to(tl.int64) * stride
    level1_row = level1_ptr + row.to(tl.int64) * groups
    group_ids = (scratch_ptr + row.to(tl.int64) * (WIDE_GROUPS // 2 + WIDE)).to(
        tl.pointer_type(tl.int32), bitcast=True
    )
    keys_ptr = scratch_ptr + row.to(tl.int64) * (WIDE_GROUPS // 2 + WIDE) + WIDE_GROUPS // 2
    row_max, row_sum = _row_stats(stats_ptr, row, SPLITS)
    target = tl.load(fallback_p_ptr + row) * row_sum
    s = tl.arange(0, SUPERGROUPS_P2)
    level2 = tl.load(level2_ptr + row.to(tl.int64) * supergroups + s, mask=s < supergroups,
                     other=-float("inf"))  # fmt: skip
    t = _wide_threshold(level1_row, level2, supergroups, row_max, target, groups,
                        SUPERGROUPS_P2, WIDE_GROUPS, BLOCK)  # fmt: skip
    if t == -float("inf"):
        tl.store(state_ptr + row, NOOP_C)
        return
    count = _compact_groups(level1_row, group_ids, t, groups, WIDE_GROUPS, BLOCK)
    tl.debug_barrier()
    found = _compact_logits(row_ptr, group_ids, keys_ptr, t, tl.minimum(count, WIDE_GROUPS),
                            vocab, WIDE, BLOCK)  # fmt: skip
    tl.debug_barrier()
    rank = tl.arange(0, WIDE)
    key = tl.sort(tl.load(keys_ptr + rank, mask=rank < found, other=0x7FFFFFFFFFFFFFFF))
    weight = tl.where(rank < found, tl.exp(_key_value(key) - row_max), 0.0)
    ok = (count <= WIDE_GROUPS) & (found <= WIDE) & (tl.sum(weight) >= target)
    keep = (rank < found) & (tl.cumsum(weight, 0) - weight < target)
    tl.debug_barrier()
    tl.store(keys_ptr + rank, key, mask=keep & ok)
    tl.store(state_ptr + row, tl.where(ok, tl.sum(keep.to(tl.int32)), NOOP_C))
    if ok:
        tl.store(fallback_p_ptr + row, 1.0)


@triton.jit
def _wide_threshold(level1_row, level2, supergroups, row_max, target, groups,
                    SUPERGROUPS_P2: tl.constexpr, WIDE_GROUPS: tl.constexpr,
                    BLOCK: tl.constexpr):  # fmt: skip
    """Of 8 supergroup maxima (ranks 1/8 to 8/8 of the supergroups), the highest whose
    group maxima alone reach the target mass (so the nucleus lies at or above it), if
    at most WIDE_GROUPS groups reach it. -inf otherwise."""
    s = tl.arange(0, SUPERGROUPS_P2)
    j = tl.arange(0, 8)
    ranks = tl.maximum((j + 1) * supergroups // 8, 1) - 1
    ordered = tl.sort(level2, descending=True)
    thresholds = tl.sum(tl.where(s[None, :] == ranks[:, None], ordered[None, :], 0.0), 1)
    counts = tl.zeros([8], tl.int32)
    masses = tl.zeros([8], tl.float32)
    offs = tl.arange(0, BLOCK // 4)
    for start in range(0, groups, BLOCK // 4):
        group = start + offs
        level1 = tl.load(level1_row + group, mask=group < groups, other=-float("inf"))
        reach = level1[None, :] >= thresholds[:, None]
        counts += tl.sum(reach.to(tl.int32), 1)
        masses += tl.sum(tl.where(reach, tl.exp(level1 - row_max)[None, :], 0.0), 1)
    pick = tl.min(tl.where(masses >= target, j, 8))
    fits = (pick < 8) & (tl.sum(tl.where(j == pick, counts, 0)) <= WIDE_GROUPS)
    return tl.where(fits, tl.sum(tl.where(j == pick, thresholds, 0.0)), -float("inf"))


@triton.jit
def _compact_groups(level1_row, group_ids, t, groups, WIDE_GROUPS: tl.constexpr,
                    BLOCK: tl.constexpr):  # fmt: skip
    """Store the ids of the groups whose max reaches t, return how many reach it."""
    offs = tl.arange(0, BLOCK)
    count = 0
    for start in range(0, groups, BLOCK):
        group = start + offs
        level1 = tl.load(level1_row + group, mask=group < groups, other=-float("inf"))
        hit = (level1 >= t).to(tl.int32)
        pos = count + tl.cumsum(hit, 0) - 1
        tl.store(group_ids + pos, group, mask=(hit > 0) & (pos < WIDE_GROUPS))
        count += tl.sum(hit)
    return count


@triton.jit
def _compact_logits(row_ptr, group_ids, keys_ptr, t, count, vocab, WIDE: tl.constexpr,
                    BLOCK: tl.constexpr):  # fmt: skip
    """Store the keys of the stored groups' logits that reach t, return how many."""
    found = 0
    for start in range(0, count, BLOCK // 16):
        slot = start + tl.arange(0, BLOCK // 16)
        group = tl.load(group_ids + slot, mask=slot < count, other=0)
        index = group[:, None] * 16 + tl.arange(0, 16)[None, :]
        x = tl.load(row_ptr + index, mask=(slot < count)[:, None] & (index < vocab),
                    other=-float("inf"))  # fmt: skip
        hit = tl.reshape((x >= t).to(tl.int32), [BLOCK])
        pos = found + tl.cumsum(hit, 0) - 1
        tl.store(keys_ptr + pos, tl.reshape(_key(x, index), [BLOCK]), mask=(hit > 0) & (pos < WIDE))
        found += tl.sum(hit)
    return found


@triton.jit
def _fill_kernel(logits_ptr, stride, state_ptr, scratch_ptr, vocab, chunk, WIDE: tl.constexpr,
                 WIDE_GROUPS: tl.constexpr, BLOCK: tl.constexpr):  # fmt: skip
    """-inf over the split's chunk, then the kept logits that fall in it."""
    row = tl.program_id(0)
    kept = tl.load(state_ptr + row)
    if kept < 0:
        return
    row_ptr = logits_ptr + row.to(tl.int64) * stride
    lo = tl.program_id(1) * chunk
    hi = tl.minimum(lo + chunk, vocab)
    offs = tl.arange(0, BLOCK)
    for start in range(lo, hi, BLOCK):
        tl.store(row_ptr + start + offs, -float("inf"), mask=start + offs < hi)
    tl.debug_barrier()
    keys_ptr = scratch_ptr + row.to(tl.int64) * (WIDE_GROUPS // 2 + WIDE) + WIDE_GROUPS // 2
    slots = tl.arange(0, 256)
    for start in range(0, kept, 256):
        key = tl.load(keys_ptr + start + slots, mask=start + slots < kept, other=0)
        index = (key & 0xFFFFFFFF).to(tl.int32)
        valid = (start + slots < kept) & (index >= lo) & (index < hi)
        tl.store(row_ptr + index, _key_value(key), mask=valid)


@triton.jit
def _row_params(k_ptr, p_ptr, row, vocab, HAS_K: tl.constexpr, HAS_P: tl.constexpr):
    k = vocab
    p = vocab.to(tl.float32) * 0.0 + 1.0
    if HAS_K:
        k = tl.load(k_ptr + row).to(tl.int32)
    if HAS_P:
        p = tl.load(p_ptr + row).to(tl.float32)
    return k, p


@triton.jit
def _row_stats(stats_ptr, row, SPLITS: tl.constexpr):
    """The row's max and sum of exp(logit - max), from the per-split partials."""
    program = row * SPLITS + tl.arange(0, SPLITS)
    split_max = tl.load(stats_ptr + 2 * program)
    split_sum = tl.load(stats_ptr + 2 * program + 1)
    row_max = tl.max(split_max)
    return row_max, tl.sum(split_sum * tl.exp(split_max - row_max))


@triton.jit
def _key(x, idx):
    """Ascending order is logit descending, then index ascending."""
    bits = tl.where(x == 0, 0.0, x).to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    ordered = tl.where(bits >= 0x80000000, 0xFFFFFFFF - bits, bits + 0x80000000)
    return ((0x7FFFFFFF - ordered) << 32) | idx.to(tl.int64)


@triton.jit
def _key_value(key):
    ordered = 0x7FFFFFFF - (key >> 32)
    bits = tl.where(ordered >= 0x80000000, ordered - 0x80000000, 0xFFFFFFFF - ordered)
    return bits.to(tl.uint32).to(tl.float32, bitcast=True)
