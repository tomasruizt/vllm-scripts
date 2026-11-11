#!/usr/bin/env bash
# Full sampler tests for the PR description. Usage, on the GPU host: tests_full.sh TREE OUT [memcheck]
set -euo pipefail
TREE=$1 OUT=$2 MEMCHECK=${3:-}
export PATH=$HOME/code/topk-cuda-main/.venv/bin:/usr/local/cuda-13.0/bin:$PATH PYTHONPATH=$TREE

main() {
    mkdir -p "$OUT"
    cd "$TREE"
    chg run --gpus 1 --timeout 120m --note topk-tests -- bash -c "$(declare -f run); run '$OUT' '$MEMCHECK'"
    echo TESTS_DONE
}

run() {
    local out=$1 memcheck=$2 f=tests/v1/sample/test_topk_topp_sampler.py
    python -m pytest $f -q -p no:cacheprovider > "$out/file.log" 2>&1 || true
    python -m pytest tests/v1/sample --deselect $f -q -p no:cacheprovider > "$out/rest.log" 2>&1 || true
    [[ -z $memcheck ]] || compute-sanitizer --tool memcheck python -m pytest $f -q -p no:cacheprovider \
        -k "TestCudaTopkToppExact and not graph" > "$out/memcheck.log" 2>&1 || true
}

main
