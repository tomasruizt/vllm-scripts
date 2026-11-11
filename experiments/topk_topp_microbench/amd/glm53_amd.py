"""glm53_bench.py for ROCm: benchmark_topk_topp.py (the ROCm copy, event timing with a
1 GiB flush) with the group-maxima kernel (topk_topp_triton_new.py) as `new`, plus
`clone-only` (returns its input, so it times the harness's clone), which the check skips."""
import benchmark_topk_topp as bench
from topk_topp_triton_new import apply_top_k_top_p_triton

bench.IMPLEMENTATIONS["new"] = apply_top_k_top_p_triton
bench.IMPLEMENTATIONS["clone-only"] = lambda logits, k, p: logits
check = bench.check_implementations


def check_filters(logits, config, implementations):
    return check(logits, config, [n for n in implementations if n != "clone-only"])


bench.check_implementations = check_filters
bench.main()
