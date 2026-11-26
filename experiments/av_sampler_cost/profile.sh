#!/usr/bin/env bash
# Torch-profile a few real steps per sampling mode on one server (AV on, policy budgets), to
# attribute the GPU time of sample() to kernels. Needs the profiler annotations of the trace patch.
# Usage: profile.sh GPU_ID   (from the run directory, inside a chg reservation)
set -euo pipefail
GPU_ID=$1
MODEL=${MODEL:-Qwen/Qwen3-4B}
SPEC_MODEL=${SPEC_MODEL:-RedHatAI/Qwen3-4B-speculator.dflash2}
METHOD=${METHOD:-dflash}
K=${K:-8}
MAX_SEQS=${MAX_SEQS:-256}
C=${C:-128}
MODES=${MODES:-greedy,temp,topp,topk_topp}
MODES=${MODES//,/ }
PROFILE_S=${PROFILE_S:-2}
VLLM_DIR=${VLLM_DIR:-$HOME/code/vllm-av-sampler-cost}
PORT=$((8900 + ${GPU_ID%%,*}))
ROOT=$PWD
export HF_HOME=/data/tomasruizt/huggingface HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda-13.0
export PATH="$VLLM_DIR/.venv/bin:$CUDA_HOME/bin:$PATH"
export AV_STEP_TRACE_PATH=$ROOT/steps.jsonl AV_STEP_TRACE_CONTROL=$ROOT/control.txt
SERVER_PID=""

main() {
  trap cleanup EXIT
  trap 'exit 143' TERM INT
  start_server
  local mode
  for mode in $MODES; do profile_mode "$mode"; done
  echo ALL_DONE
}

profile_mode() {  # <sampling mode>: load for 60s, profile PROFILE_S seconds at steady state
  local mode=$1 load_pid
  echo "policy main $mode c$C" > control.txt
  echo "CELL $mode $(date -u +%FT%TZ)"
  load "$mode" & load_pid=$!
  sleep 35  # AIPerf startup plus ramp-up, then every request slot is busy
  echo "policy profiled $mode c$C" > control.txt
  curl -sf -X POST "http://127.0.0.1:$PORT/start_profile"
  sleep "$PROFILE_S"
  curl -sf -X POST "http://127.0.0.1:$PORT/stop_profile"
  echo "policy main $mode c$C" > control.txt
  wait "$load_pid" || echo "aiperf $mode exit $?" >> failures.txt
  mkdir -p "traces/$mode" && mv trace-tmp/* "traces/$mode/"
}

start_server() {
  git -C "$VLLM_DIR" log -2 --format='%H %cs %s' > commit.txt
  local spec="{\"model\":\"$SPEC_MODEL\",\"num_speculative_tokens\":$K,\"method\":\"$METHOD\",\"enable_adaptive_verification\":true}"
  local profiler="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$ROOT/trace-tmp\",\"torch_profiler_with_stack\":false,\"torch_profiler_dump_cuda_time_total\":false}"
  local args=(vllm serve "$MODEL" --port "$PORT" --max-model-len 16384 --max-num-seqs "$MAX_SEQS"
    --max-num-batched-tokens 16384 --gpu-memory-utilization 0.85 --no-enable-prefix-caching
    --disable-uvicorn-access-log --speculative-config "$spec" --profiler-config "$profiler"
    --default-chat-template-kwargs '{"enable_thinking": true}'
    --compilation-config "{\"max_cudagraph_capture_size\": $((MAX_SEQS * (K + 1)))}" ${SERVE_ARGS:-})
  printf '%q ' "${args[@]}" > serve-command.txt
  "${args[@]}" > server.log 2>&1 &
  SERVER_PID=$!
  timeout 1800 bash -c "until curl -sf http://127.0.0.1:$PORT/health >/dev/null; do kill -0 $SERVER_PID || exit 1; sleep 5; done"
}

load() {  # <sampling mode>
  local extra
  case $1 in
    greedy)    extra='"temperature":0' ;;
    temp)      extra='"temperature":0.7,"top_p":1,"top_k":-1' ;;
    topp)      extra='"temperature":0.7,"top_p":0.8,"top_k":-1' ;;
    topk_topp) extra='"temperature":0.7,"top_p":0.8,"top_k":20' ;;
    *) echo "unknown mode $1" >&2; exit 1 ;;
  esac
  timeout 600 "$HOME/.venv/bin/aiperf" profile --model "$MODEL" --tokenizer "${TOKENIZER:-$MODEL}" --url "http://127.0.0.1:$PORT" \
    --endpoint-type chat --streaming --input-file "$ROOT/prompts/mixed.jsonl" \
    --custom-dataset-type single_turn --dataset-sampling-strategy sequential \
    --concurrency "$C" --benchmark-duration 50 --benchmark-grace-period 60 \
    --extra-inputs "{\"max_tokens\":512,$extra}" \
    --artifact-dir "aiperf/$1" --ui-type none > "aiperf-$1.log" 2>&1
}

cleanup() {
  local code=$?
  [[ -z $SERVER_PID ]] || { kill "$SERVER_PID" || true; wait "$SERVER_PID" || true; }
  echo "$code" > exit-code.txt
}

main
