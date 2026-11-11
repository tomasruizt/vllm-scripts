#!/usr/bin/env bash
# #65649 description numbers: its own benchmark, Triton vs. the FlashInfer fast path, 3 passes
# (implementation order reversed in the second), two k/p profiles. Usage, on the GPU host:
# pr65649_bench.sh TREE OUT
set -euo pipefail
TREE=$1 OUT=$2
export PATH=$HOME/code/topk-cuda-main/.venv/bin:$PATH PYTHONPATH=$TREE

main() {
    mkdir -p "$OUT"
    cd "$TREE"
    chg run --gpus 1 --timeout 180m --note topk-pr65649 -- bash -c "$(declare -f run one); run '$OUT'"
    echo BENCH_DONE
}

run() {
    local out=$1 pass impls
    for pass in 0 1 2; do
        impls="triton flashinfer-fastpath"
        [[ $pass != 1 ]] || impls="flashinfer-fastpath triton"
        one "$out" "$pass" 20 0.8 "$impls"
        one "$out" "$pass" 100 0.95 "$impls"
    done
}

one() {
    local out=$1 pass=$2 k=$3 p=$4 impls=$5
    python benchmarks/benchmark_topk_topp.py --batch-sizes 1 4 8 16 64 128 256 512 1024 2048 \
        --vocab-sizes 65536 131072 151936 248320 --scenarios topk_whole topk_topp_whole \
        --top-k "$k" --top-p "$p" --distribution zipf --implementations $impls --check \
        --warmup-iters 5 --benchmark-iters 30 --output "$out/k$k-pass-$pass.jsonl" > "$out/k$k-pass-$pass.log" 2>&1
}

main "$@"
