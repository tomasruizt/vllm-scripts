#!/usr/bin/env bash
# Kernel validation on one GPU: GDN tests on the PR and split-V builds, compute-sanitizer on the
# split-V kernel, then the microbenchmark of both builds. Run from the run directory inside chg.
set -uo pipefail

CODE=${CODE:-$HOME/code}
MAIN=${MAIN:-$CODE/gdn-k8-verify-b300-main}
TESTS=(tests/kernels/fla/test_fused_gdn_post_conv.py tests/kernels/mamba/test_gdn_fused_mtp.py
    tests/model_executor/test_qwen_triton_warmup.py)
SANITIZER_TESTS="mtp and cuda"
LABELS=${LABELS:-pr splitv}  # the last label is the one under compute-sanitizer
BENCH_LABELS=${BENCH_LABELS:-$LABELS}  # may include main
BENCH_TOKENS=${BENCH_TOKENS:-9}  # tokens per request: K+1, e.g. 4 for MTP3
export PATH=$MAIN/.venv/bin:$PATH:/usr/local/cuda/bin VLLM_USE_V2_MODEL_RUNNER=1

main() {
    local label
    for label in $LABELS; do run_tests "$label"; done
    for tool in memcheck racecheck synccheck; do sanitize "$tool"; done
    for label in $BENCH_LABELS; do for tokens in $BENCH_TOKENS; do bench "$label" "$tokens"; done; done
    echo VALIDATE_DONE
}

source_dir() {
    case $1 in main) echo "$MAIN" ;; pr) echo "$CODE/gdn-k8-pr-on-main" ;; *) echo "$CODE/gdn-k8-$1" ;; esac
}

run_tests() {
    local label=$1 dir
    dir=$(source_dir "$label")
    (cd "$dir" && PYTHONPATH=$dir python -m pytest -q -p no:cacheprovider "${TESTS[@]}") > "tests-$label.log" 2>&1
    echo "tests $label: $(tail -n 1 "tests-$label.log")"
}

sanitize() {
    local tool=$1 dir
    dir=$(source_dir "${LABELS##* }")
    (cd "$dir" && PYTHONPATH=$dir compute-sanitizer --tool "$tool" --kernel-name kns=gdn_decode_post_conv_mtp \
        --error-exitcode 99 python -m pytest -q -p no:cacheprovider "${TESTS[0]}" -k "$SANITIZER_TESTS") \
        > "sanitizer-$tool.log" 2>&1
    echo "sanitizer $tool: exit $? $(grep -E 'ERROR SUMMARY|RACECHECK SUMMARY' "sanitizer-$tool.log" | tail -n 1)"
}

bench() {
    local label=$1 tokens=$2 dir out=bench-$1-t$2
    dir=$(source_dir "$label")
    PYTHONPATH=$dir python kernel_bench.py --label "$label" --tokens "$tokens" > "$out.jsonl" 2> "$out.log"
    echo "bench $label tokens=$tokens: $(wc -l < "$out.jsonl") cells"
}

main "$@"
