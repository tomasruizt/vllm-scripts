#!/usr/bin/env bash
# 3 passes of the FlashInfer fast path variants (fi_opt.py) vs. CUDA and Triton, on the
# scenarios the fast path supports. Usage, on the GPU host: fi_opt_bench.sh TREE OUT IMPLS...
set -euo pipefail
TREE=$1 OUT=$2
shift 2
export PATH=$HOME/code/topk-cuda-main/.venv/bin:$PATH PYTHONPATH=$TREE:$HOME/code/topk-research

main() {
    mkdir -p "$OUT"
    cd "$TREE"
    chg run --gpus 1 --timeout 180m --note topk-fiopt -- bash -c "$(declare -f run); run '$OUT' $*"
    echo BENCH_DONE
}

run() {
    local out=$1 pass
    shift
    for pass in 0 1 2; do
        python "$HOME/code/topk-research/fi_opt.py" "$out/pass-$pass.jsonl" \
            --logits-file "$HOME/code/qwen36-verify-logits-t1.pt" --implementations "$@" > "$out/pass-$pass.log" 2>&1
    done
}

main "$@"
