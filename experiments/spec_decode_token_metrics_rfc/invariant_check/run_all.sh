#!/usr/bin/env bash
# Runs all invariant-check configs in parallel, one GPU each.
# Usage: run_all.sh (from the vLLM venv, with HF_HOME set).
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ngram='{"method": "ngram_gpu", "num_speculative_tokens": 4, "prompt_lookup_min": 1, "prompt_lookup_max": 3}'
eagle='{"method": "eagle3", "model": "RedHatAI/Qwen3-8B-speculator.eagle3", "num_speculative_tokens": 3}'
dspark_av='{"method": "dspark", "model": "RedHatAI/Qwen3-4B-speculator.dspark", "num_speculative_tokens": 7, "enable_adaptive_verification": true}'
dspark='{"method": "dspark", "model": "RedHatAI/Qwen3-4B-speculator.dspark", "num_speculative_tokens": 7}'

run() {  # name port model spec mode [extra] [env...]
  local name=$1; shift
  env "${@:6}" chg run --gpus 1 --timeout 40m --note "sd-invariant-$name" -- \
    bash "$here/run_one.sh" "$name" "$@" 2>&1 | tail -n 1 &
}

run ngram_gpu_sync 8101 Qwen/Qwen3-0.6B "$ngram" sync ""
run ngram_gpu_async 8102 Qwen/Qwen3-0.6B "$ngram" async ""
run eagle3_sync 8103 Qwen/Qwen3-8B "$eagle" sync ""
run eagle3_async 8104 Qwen/Qwen3-8B "$eagle" async ""
run dspark_async 8105 Qwen/Qwen3-4B "$dspark" async ""
run dspark_av_sync 8106 Qwen/Qwen3-4B "$dspark_av" sync --expect-av
run dspark_av_async 8107 Qwen/Qwen3-4B "$dspark_av" async --expect-av
run dspark_av_async_rust 8108 Qwen/Qwen3-4B "$dspark_av" async --expect-av VLLM_USE_RUST_FRONTEND=1
wait
echo ALL_DONE
