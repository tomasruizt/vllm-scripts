#!/usr/bin/env bash
# A/B of kernel source variants in one scratch worktree: per variant, swap in SRC_DIR/v-<variant>.cu as the fused
# GDN kernel, rebuild _C_stable_libtorch incrementally, run kernel_bench.py (and optional TESTS). Run inside chg.
# Usage: VARIANTS="main pr unroll" TOKENS="4 9" [TESTS="tests/kernels/..."] kernel_variants.sh
# Never run it while a server uses the scratch worktree, the install rewrites its .so.
set -euo pipefail
MAIN=${MAIN:-$HOME/code/gdn-k8-verify-b300-main} WT=${WT:-$HOME/code/gdn-k8-ab} SRC_DIR=${SRC_DIR:-cc}
export PATH=$MAIN/.venv/bin:$PATH:/usr/local/cuda-13.0/bin TORCH_CUDA_ARCH_LIST=${ARCH:-10.3}

main() {
    local v
    for v in ${VARIANTS:?}; do run_variant "$v"; done
    git -C "$WT" checkout -- csrc/gdn/fused_gdn_decode_kernel.cu
    echo VARIANTS_DONE
}

run_variant() {
    local v=$1 t
    cp "$SRC_DIR/v-$v.cu" "$WT/csrc/gdn/fused_gdn_decode_kernel.cu"
    (cd "$WT" && cmake --build cmake-build-release --target _C_stable_libtorch -j 64 \
        && cmake --install cmake-build-release --component _C_stable_libtorch) > "build-$v.log" 2>&1
    for t in ${TOKENS:-4}; do
        PYTHONPATH=$WT python kernel_bench.py --label "$v" --impls cuda --tokens "$t" --batches 1 2 4 8 16 32 64 \
            --repeats 5 > "bench-$v-t$t.jsonl" 2> "bench-$v-t$t.log"
    done
    [[ -z ${TESTS:-} ]] || (cd "$WT" && PYTHONPATH=$WT python -m pytest -q -p no:cacheprovider $TESTS) > "tests-$v.log" 2>&1 || true
    echo "$v: $(cat bench-"$v"-t*.jsonl | wc -l) cells, tests: $(tail -n 1 "tests-$v.log" 2>/dev/null)"
}

main "$@"
