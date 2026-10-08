#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/env.sh"
export VLLM_USE_V2_MODEL_RUNNER=${1:?Specify runner 0 or 1}
run_name=api
compilation_config='{"cudagraph_capture_sizes":[8,16,32,64,128,256,512]}'
if [[ ${HS_GRAPH_MODE:-} == PIECEWISE ]]; then
    run_name=api-piecewise
    compilation_config='{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[8,16,32,64,128,256,512]}'
fi
export HS_OUTPUT_DIR=${HS_OUTPUT_DIR:-$HS_RUNS_DIR/$run_name-runner${VLLM_USE_V2_MODEL_RUNNER}}
export HS_PORT=$((8150 + VLLM_USE_V2_MODEL_RUNNER))
export HS_ENDPOINT=http://127.0.0.1:$HS_PORT
export PYTHONPATH=$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}
export VLLM_SERVER_DEV_MODE=1
if [[ ${2:-} != reserved ]]; then
    exec canhazgpu run --gpus "$TP_SIZE" --wait 30m --timeout 90m --note "glm-hs-speculators-api-runner$VLLM_USE_V2_MODEL_RUNNER" -- bash "$0" "$1" reserved
fi
cd "$VLLM_DIR"
mkdir -p "$HS_OUTPUT_DIR"
test ! -e "$HS_OUTPUT_DIR/oracle/manifest.jsonl"
python=/home/tomasruizt/.venv/bin/python
"$python" /home/tomasruizt/code/speculators/scripts/launch_vllm.py train "$MODEL_DIR" \
    --target-layer-ids 5 22 43 --hidden-states-path "$HS_OUTPUT_DIR/saved" \
    --provenance-dir "$HS_OUTPUT_DIR" --no-hash-checkpoints -- \
    --host 127.0.0.1 --port "$HS_PORT" --api-server-count 1 --renderer-num-workers 1 \
    --tensor-parallel-size "$TP_SIZE" --dtype bfloat16 \
    --max-model-len 4096 --max-num-batched-tokens 512 --max-num-seqs 4 \
    --gpu-memory-utilization 0.85 --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image":0,"video":0}' --disable-uvicorn-access-log \
    --compilation-config "$compilation_config" \
    --worker-extension-cls api_oracle.ApiOracleExtension > "$HS_OUTPUT_DIR/server.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true' EXIT
ready=0
for ((attempt=0; attempt<1800; attempt++)); do
    if ! kill -0 "$server" 2>/dev/null; then tail -80 "$HS_OUTPUT_DIR/server.log"; exit 1; fi
    if curl --silent --fail "$HS_ENDPOINT/health" >/dev/null; then ready=1; break; fi
    sleep 2
done
test "$ready" = 1
"$python" "$SCRIPT_DIR/validate_api.py" 2>&1 | tee "$HS_OUTPUT_DIR/client.log"
