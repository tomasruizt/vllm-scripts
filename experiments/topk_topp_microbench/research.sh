#!/usr/bin/env bash
# Path counters on a build with stats.patch, then Triton vs. CUDA microbenchmarks on
# the clean build (synthetic, real logits, large k). Usage, on the GPU host: research.sh OUT
set -euo pipefail
OUT=$1
ROOT=$HOME/code
TREE=$ROOT/topk-cuda-pr2
EXP=$ROOT/topk-research
LOGITS=$ROOT/qwen36-verify-logits-t1.pt
BATCHES="1 4 8 16 64 128 256 512 1024 2048"
export PATH=$ROOT/topk-cuda-main/.venv/bin:/usr/local/cuda-13.0/bin:$PATH PYTHONPATH=$TREE:$EXP

main() {
    [[ -n ${BENCH_ONLY:-} ]] && { cd "$TREE"; gpu bash -c "$(declare -f bench); SKIP_LARGE_K=1 bench '$OUT'"; echo BENCH_DONE; return; }
    cd "$TREE"
    mkdir -p "$OUT"
    cp csrc/topk_topp_filter.cu "$OUT/clean.cu"
    patch -s -p1 < "$EXP/stats.patch"
    build
    gpu python "$EXP/path_counts.py" "$OUT/path-counts.jsonl" --logits-file "$LOGITS"
    cp "$OUT/clean.cu" csrc/topk_topp_filter.cu
    build
    gpu bash -c "$(declare -f bench); bench '$OUT'"
    echo BENCH_DONE
}

build() {
    cmake --build cmake-build-release --target _C_stable_libtorch -j 32 > "$OUT/build.log" 2>&1
    cmake --install cmake-build-release --component _C_stable_libtorch >> "$OUT/build.log"
}

gpu() { chg run --gpus 1 --timeout 180m --note topk-research -- "$@"; }

bench() {
    local out=$1 pass
    for pass in 0 1 2; do
        python benchmarks/benchmark_topk_topp.py --batch-sizes 1 4 8 16 64 128 256 512 1024 2048 \
            --vocab-sizes 131072 248320 --logits-file "$HOME/code/qwen36-verify-logits-t1.pt" \
            --implementations triton cuda --check --output "$out/pass-$pass.jsonl" > "$out/pass-$pass.log" 2>&1
        [[ -n ${SKIP_LARGE_K:-} ]] || python "$HOME/code/topk-research/large_k.py" "$out/large-k-pass-$pass.jsonl" > "$out/large-k-pass-$pass.log" 2>&1
    done
}

main "$@"
