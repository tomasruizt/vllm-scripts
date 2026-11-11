"""benchmarks/benchmark_topk_topp.py with the group-maxima Triton kernel (a copy of the
vllm-topk-triton-hier worktree's topk_topp_triton.py as topk_topp_triton_new) as impl `new`."""
import benchmark_topk_topp as bench
from topk_topp_triton_new import apply_top_k_top_p_triton

bench.IMPLEMENTATIONS["new"] = apply_top_k_top_p_triton
bench.main()
