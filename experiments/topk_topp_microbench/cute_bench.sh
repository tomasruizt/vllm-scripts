#!/usr/bin/env bash
# CuTe DSL vs. CUDA filter (rh-xhuu), on the GPU host from its own tree: CUDA filter tests for both
# variants, startup cost (cold and warm JIT cache), 3 passes of Triton/CUDA/CuTe in rotating order
# (synthetic V131k/V248k and real logits), and large k. Usage: cute_bench.sh OUT
set -euo pipefail
OUT=$1
ROOT=$HOME/code
TREE=${TREE:-$ROOT/topk-cute}
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH=$ROOT/topk-cuda-main/.venv/bin:$PATH PYTHONPATH=$TREE

main() {
    mkdir -p "$OUT"
    chg run --gpus 1 --timeout 300m --note topk-cute-final -- bash -c "$(declare -f run startup); run '$OUT' '$HERE' '$TREE'"
    echo BENCH_DONE
}

run() {
    local out=$1 here=$2 tree=$3 pass
    cd "$tree"
    python -m pytest tests/v1/sample/test_topk_topp_sampler.py -q -p no:cacheprovider -k Cuda > "$out/tests.log" 2>&1 || true
    startup "$out" "$here"
    local orders=("triton cuda cute" "cute triton cuda" "cuda cute triton")
    for pass in 0 1 2; do
        python benchmarks/benchmark_topk_topp.py --batch-sizes 1 4 8 16 64 128 256 512 1024 2048 \
            --vocab-sizes 131072 248320 --logits-file "$HOME/code/qwen36-verify-logits-t1.pt" \
            --implementations ${orders[$pass]} --check --output "$out/pass-$pass.jsonl" > "$out/pass-$pass.log" 2>&1
        python "$here/large_k.py" "$out/large-k-$pass.jsonl" --implementations ${orders[$pass]} > "$out/large-k-$pass.log" 2>&1
    done
}

startup() {
    local out=$1 here=$2 cache
    cache=$(mktemp -d)
    CUTE_DSL_CACHE_DIR=$cache python "$here/cute_startup.py" "$out/startup-cold.json" > "$out/startup.log" 2>&1
    CUTE_DSL_CACHE_DIR=$cache python "$here/cute_startup.py" "$out/startup-warm.json" >> "$out/startup.log" 2>&1
    rm -rf "$cache"
}

main "$@"
