#!/usr/bin/env bash
# Rebuild the PR tree, run the CUDA filter tests, then 3 passes of Triton vs. CUDA
# (synthetic V131k/V248k and real logits). Usage, on the GPU host: final_bench.sh TREE OUT
set -euo pipefail
TREE=$1 OUT=$2
ROOT=$HOME/code
export PATH=$ROOT/topk-cuda-main/.venv/bin:/usr/local/cuda-13.0/bin:$PATH PYTHONPATH=$TREE

main() {
    cd "$TREE"
    mkdir -p "$OUT"
    cmake --build cmake-build-release --target _C_stable_libtorch -j 32 > "$OUT/build.log" 2>&1
    cmake --install cmake-build-release --component _C_stable_libtorch >> "$OUT/build.log"
    chg run --gpus 1 --timeout 240m --note topk-final -- bash -c "$(declare -f run); run '$OUT'"
    echo BENCH_DONE
}

run() {
    local out=$1 pass
    python -m pytest tests/v1/sample/test_topk_topp_sampler.py -q -p no:cacheprovider -k Cuda > "$out/tests.log" 2>&1 || true
    tail -n 1 "$out/tests.log"
    for pass in 0 1 2; do
        python benchmarks/benchmark_topk_topp.py --batch-sizes 1 4 8 16 64 128 256 512 1024 2048 \
            --vocab-sizes 131072 248320 --logits-file "$HOME/code/qwen36-verify-logits-t1.pt" \
            --implementations triton cuda --check --output "$out/pass-$pass.jsonl" > "$out/pass-$pass.log" 2>&1
    done
}

main "$@"
