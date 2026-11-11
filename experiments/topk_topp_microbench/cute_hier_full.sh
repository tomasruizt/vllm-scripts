#!/usr/bin/env bash
# Full suite for the CuTe hierarchical kernel (rh-4spj), on the GPU host from the CuTe tree, after the
# gate passed: the CUDA filter tests for cute-hier, JIT compile time, then 3 passes in rotating
# implementation order of the benchmark_topk_topp.py configs (synthetic V131k/V248k, real logits)
# and of large k. Usage: cute_hier_full.sh OUT   (writes OUT/done.txt when finished)
set -euo pipefail
OUT=$1
ROOT=$HOME/code
TREE=${TREE:-$ROOT/topk-cute}
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH=$ROOT/topk-cuda-main/.venv/bin:$PATH PYTHONPATH=$TREE:$HERE

main() {
    mkdir -p "$OUT"
    chg run --gpus 1 --timeout 240m --note topk-cute-hier-full -- bash -c "$(declare -f run); run '$OUT' '$HERE' '$TREE'" > "$OUT/run.log" 2>&1
    echo $? > "$OUT/done.txt"
}

run() {
    local out=$1 here=$2 tree=$3 pass logits=$HOME/code/qwen36-verify-logits-t1.pt
    cd "$tree"
    python -m pytest tests/v1/sample/test_topk_topp_sampler.py -q -p no:cacheprovider -k "Cuda and hier" > "$out/tests.log" 2>&1 || true
    python -c "import time, torch; torch.zeros(1, device='cuda'); t = time.perf_counter()
from vllm.v1.sample.ops.topk_topp_hier import reserve_topk_topp_hier_workspace as r; r(torch.device('cuda')); print('compile_s', round(time.perf_counter() - t, 1))" > "$out/startup.log" 2>&1
    local orders=("triton cuda cute cute-hier hier-cuda" "cute-hier hier-cuda triton cuda cute" "hier-cuda cute triton cute-hier cuda")
    for pass in 0 1 2; do
        python "$here/cute_hier_bench.py" "$out/pass-$pass.jsonl" --logits-file "$logits" --implementations ${orders[$pass]} > "$out/pass-$pass.log" 2>&1
        python "$here/cute_hier_bench.py" "$out/large-k-$pass.jsonl" --logits-file "$logits" --large-k --implementations ${orders[$pass]} > "$out/large-k-$pass.log" 2>&1
    done
}

main "$@"
