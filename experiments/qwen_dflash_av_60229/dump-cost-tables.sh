#!/usr/bin/env bash
# Start one AV-on server (same config as run-steady.sh) with debug logging, wait for the
# "DSpark cost tables" line, save it, and stop the server. Usage: dump-cost-tables.sh <gpu-ids> <out_dir>
# Optional env: TP (default 1), MAX_NUM_SEQS (default 512), CG_SIZES (comma list, default: vLLM's).
set -euo pipefail
gpu=$1 out=$2
extra=()
[[ -n "${CG_SIZES:-}" ]] && extra=(--compilation-config "{\"cudagraph_capture_sizes\":[$CG_SIZES]}")
mkdir -p "$out/site"
# Print full numpy arrays, so the 16K-entry verify table is not shortened with "...".
echo 'import sys, numpy; numpy.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)' > "$out/site/sitecustomize.py"
export PATH="$HOME/.local/bin:$HOME/.venv/bin:$PATH"
export PYTHONPATH="$out/site:$HOME/code/vllm-validate-60229" VLLM_LOGGING_LEVEL=DEBUG
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=1024
spec='{"method":"dflash","model":"z-lab/Qwen3.5-35B-A3B-DFlash","revision":"52cb554b4995dede3e2e1bdb129cdb1f3529332b","num_speculative_tokens":7,"draft_sample_method":"probabilistic","enable_adaptive_verification":true}'
canhazgpu run --gpu-ids "$gpu" --timeout 30m --note "AV cost table dump" -- setsid "$HOME/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
    --model /data/tomasruizt/hub_cache/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754 \
    --served-model-name qwen --language-model-only --max-model-len 16384 \
    --tensor-parallel-size "${TP:-1}" --max-num-seqs "${MAX_NUM_SEQS:-512}" --max-num-batched-tokens 16384 --gpu-memory-utilization 0.92 \
    --no-enable-prefix-caching --disable-uvicorn-access-log --host 127.0.0.1 --port 8640 \
    --speculative-config "$spec" "${extra[@]}" > "$out/server.log" 2>&1 &
timeout 900 bash -c "until grep -q 'DSpark cost tables' '$out/server.log' 2>/dev/null; do sleep 5; done"
grep -m1 'DSpark cost tables' "$out/server.log" > "$out/cost-tables.txt"
grep -E 'cudagraph_capture_sizes|KV cache size' "$out/server.log" | head -2 > "$out/server-info.txt" || true
pkill -TERM -f 'api_server.*--port 864[0]' || true
echo DONE
