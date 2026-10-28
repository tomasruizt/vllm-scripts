#!/usr/bin/env bash
# Invoked inside canhazgpu, with its CUDA_VISIBLE_DEVICES reservation.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root="${DSV41_ROOT:-$HOME/runs/dsv41-cost-20261021}"
envdir="$HOME/venvs/dsv41-cost-20261021"
model=${DSV41_MODEL:-/data/engine/hub_cache/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170}
port=${DSV41_PORT:-8681}
tp=4
requests=256
memory=0.92
extra=()
if [[ "${DSV41_SMOKE:-0}" == 1 ]]; then
    tp=1
    requests=8
    memory=0.35
    extra=(--load-format dummy --tokenizer /data/engine/hub_cache/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 --no-enable-flashinfer-autotune)
fi
server_pid=""
telemetry_pid=""
export PATH="$envdir/bin:/usr/local/cuda-13.0/bin:$HOME/.local/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-13.0
export PYTHONPATH="$here:$HOME/code/vllm-scripts" FULL_CURVE_ROOT="$root" FULL_CURVE_MAX_TARGET_TOKENS=16384
export VLLM_USE_V2_MODEL_RUNNER=1 OMP_NUM_THREADS=1
unset VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN
cd "$root"

cleanup() {
    local code=$?
    trap - EXIT
    [[ -z "$telemetry_pid" ]] || kill "$telemetry_pid" 2>/dev/null || true
    if [[ -n "$server_pid" ]]; then
        kill -TERM -- "-$server_pid" 2>/dev/null || true
        timeout 60 tail --pid="$server_pid" -f /dev/null || kill -KILL -- "-$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    echo "$code" > "$root/exit-code.txt"
}
trap cleanup EXIT
trap 'exit 143' TERM INT
if curl --silent --max-time 2 "http://127.0.0.1:$port/health" >/dev/null; then
    echo "Port $port is already occupied" >&2
    exit 1
fi
printf '{"mode":"calibrate","phase":"startup"}\n' > control.json
printf '%s\n' "$CUDA_VISIBLE_DEVICES" > gpus.txt
env | sort | grep -E '^(CUDA_VISIBLE_DEVICES|CUDA_HOME|VLLM_|FULL_CURVE_|OMP_NUM_THREADS|PYTHONPATH)=' > server-env.txt
python -c 'import vllm; print(vllm.__file__)' > preflight/vllm-path.txt
captures=$(printf '%s\n' 1 2 4 $(seq 8 8 248) $(seq 256 16 1024) $(seq 1280 256 4096) 5120 6144 | paste -sd,)
command=(python -m vllm.entrypoints.openai.api_server
    --model "$model" --served-model-name deepseek --language-model-only
    --tensor-parallel-size "$tp" --enable-expert-parallel
    --max-model-len 32768 --max-num-seqs "$requests" --max-num-batched-tokens 16384
    --gpu-memory-utilization "$memory" --no-enable-prefix-caching
    --disable-uvicorn-access-log --host 127.0.0.1 --port "$port"
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enable_adaptive_verification":true}'
    --compilation-config '{"cudagraph_capture_sizes":['"$captures"']}' "${extra[@]}")
printf '%q ' "${command[@]}" > server-command.txt
printf '\n' >> server-command.txt
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,uuid,temperature.gpu,power.draw,clocks.sm,utilization.gpu,memory.used --format=csv -l 1 > gpu-telemetry.csv &
telemetry_pid=$!
setsid "${command[@]}" > server.log 2>&1 &
server_pid=$!
export server_pid port
timeout 1800 bash -c 'until curl --silent --fail --max-time 2 "http://127.0.0.1:$port/health" >/dev/null; do kill -0 "$server_pid" || exit 1; sleep 2; done'
echo SERVER_READY
"$HOME/.venv/bin/python" "$here/run_dsv41_full.py" "$root" "$port" "$model"
