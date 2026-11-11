#!/usr/bin/env bash
# Forced splits (TOPK_SPLITS env, research build) vs. the clean heuristic around the
# cluster threshold, CUDA only. Usage, on the GPU host: sweep_splits.sh OUT
set -euo pipefail
OUT=$1
ROOT=$HOME/code
TREE=$ROOT/topk-cuda-pr2
export PATH=$ROOT/topk-cuda-main/.venv/bin:/usr/local/cuda-13.0/bin:$PATH PYTHONPATH=$TREE

main() {
    cd "$TREE"
    mkdir -p "$OUT"
    cp csrc/topk_topp_filter.cu "$OUT/clean.cu"
    cp vllm/_C_stable_libtorch.abi3.so "$OUT/clean.so"
    sed -i 's/^  const int splits = tf::choose_splits(pl, int(batch), int(vocab));$/  const char* forced = getenv("TOPK_SPLITS");\n  const int splits = forced ? atoi(forced) : tf::choose_splits(pl, int(batch), int(vocab));/' csrc/topk_topp_filter.cu
    grep -q TOPK_SPLITS csrc/topk_topp_filter.cu
    cmake --build cmake-build-release --target _C_stable_libtorch -j 32 > "$OUT/build.log" 2>&1
    cmake --install cmake-build-release --component _C_stable_libtorch >> "$OUT/build.log"
    cp "$OUT/clean.cu" csrc/topk_topp_filter.cu
    chg run --gpus 1 --timeout 120m --note topk-sweep-splits -- bash -c "$(declare -f bench); bench '$OUT'"
    cp "$OUT/clean.so" vllm/_C_stable_libtorch.abi3.so
    touch csrc/topk_topp_filter.cu
    echo BENCH_DONE
}

bench() {
    local out=$1 pass splits
    for pass in 0 1 2; do
        for splits in auto 1 2 4; do
            env $([[ $splits == auto ]] || echo TOPK_SPLITS=$splits) python benchmarks/benchmark_topk_topp.py \
                --batch-sizes 96 128 160 192 224 256 296 384 --vocab-sizes 131072 248320 \
                --logits-file "$HOME/code/qwen36-verify-logits-t1.pt" --implementations cuda \
                --output "$out/s$splits-pass-$pass.jsonl" > "$out/s$splits-pass-$pass.log" 2>&1
        done
    done
}

main "$@"
