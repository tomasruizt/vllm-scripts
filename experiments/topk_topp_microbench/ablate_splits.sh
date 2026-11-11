#!/usr/bin/env bash
# Cluster ablation: the clean build vs. one CTA per row (choose_splits returns 1),
# CUDA only, alternating per pass. Usage, on the GPU host: ablate_splits.sh OUT
set -euo pipefail
OUT=$1
ROOT=$HOME/code
TREE=$ROOT/topk-cuda-pr2
SO=vllm/_C_stable_libtorch.abi3.so
export PATH=$ROOT/topk-cuda-main/.venv/bin:/usr/local/cuda-13.0/bin:$PATH PYTHONPATH=$TREE

main() {
    cd "$TREE"
    mkdir -p "$OUT"
    cp csrc/topk_topp_filter.cu "$OUT/clean.cu"
    sed -i 's/^int choose_splits(const Plan& pl, int batch, int vocab) {$/&\n  return 1;/' csrc/topk_topp_filter.cu
    build splits1
    cp "$OUT/clean.cu" csrc/topk_topp_filter.cu
    build clean
    chg run --gpus 1 --timeout 120m --note topk-ablate-splits -- bash -c "$(declare -f bench); bench '$OUT'"
    cp "$OUT/clean.so" "$SO"
    echo BENCH_DONE
}

build() {
    cmake --build cmake-build-release --target _C_stable_libtorch -j 32 > "$OUT/build-$1.log" 2>&1
    cmake --install cmake-build-release --component _C_stable_libtorch >> "$OUT/build-$1.log"
    cp vllm/_C_stable_libtorch.abi3.so "$OUT/$1.so"
}

bench() {
    local out=$1 pass variant
    for pass in 0 1 2; do
        for variant in clean splits1; do
            cp "$out/$variant.so" vllm/_C_stable_libtorch.abi3.so
            python benchmarks/benchmark_topk_topp.py --batch-sizes 1 4 8 16 64 128 256 512 \
                --vocab-sizes 131072 248320 --logits-file "$HOME/code/qwen36-verify-logits-t1.pt" \
                --implementations cuda --output "$out/$variant-pass-$pass.jsonl" > "$out/$variant-pass-$pass.log" 2>&1
        done
    done
}

main "$@"
