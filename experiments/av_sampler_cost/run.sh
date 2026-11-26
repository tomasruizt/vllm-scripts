#!/usr/bin/env bash
# One GPU, one server with AV on and the step trace patch. Sweeps sampling modes x concurrencies.
# Every cell runs twice: with a uniform random AV budget per step (row-count sweep), then with the policy budget.
# Usage: run.sh GPU_ID   (from the run directory, inside a chg reservation)
set -euo pipefail
GPU_ID=$1
MODEL=${MODEL:-Qwen/Qwen3-4B}
SPEC_MODEL=${SPEC_MODEL:-RedHatAI/Qwen3-4B-speculator.dflash2}
METHOD=${METHOD:-dflash}
TOKENIZER=${TOKENIZER:-$MODEL}  # for AIPerf, a local path if the hub cache lacks the default revision
K=${K:-8}
MAX_SEQS=${MAX_SEQS:-256}
CONCURRENCIES=${CONCURRENCIES:-32,128,256}  # comma-separated, like MODES
CONCURRENCIES=${CONCURRENCIES//,/ }
MODES=${MODES:-greedy,temp,topp,topk_topp,penalties,structured,mixed}
MODES=${MODES//,/ }
BUDGETS=${BUDGETS:-random,policy}  # sampler (on the vllm#62960 merge): policy plus the measured sample() time
BUDGETS=${BUDGETS//,/ }
REPEATS=${REPEATS:-1}
CELL_S=${CELL_S:-20}
VLLM_DIR=${VLLM_DIR:-$HOME/code/vllm-av-sampler-cost}
PORT=$((8800 + ${GPU_ID%%,*}))
ROOT=$PWD
export HF_HOME=/data/tomasruizt/huggingface HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda-13.0
export PATH="$VLLM_DIR/.venv/bin:$CUDA_HOME/bin:$PATH"
export AV_STEP_TRACE_PATH=$ROOT/steps.jsonl AV_STEP_TRACE_CONTROL=$ROOT/control.txt
SERVER_PID=""

main() {
  trap cleanup EXIT
  trap 'exit 143' TERM INT
  start_server
  local c mode budget
  for mode in $MODES; do run_cell policy warmup "$mode" 8 5; done  # JIT every sampling path once
  local rep
  if [[ " $BUDGETS " == *" sampler "* ]]; then  # random budgets span the row counts for the sample() time fit
    for c in $CONCURRENCIES; do for mode in $MODES; do run_cell random seed "$mode" "$c" 10; done; done
  fi
  for rep in $(seq "$REPEATS"); do for c in $CONCURRENCIES; do for mode in $MODES; do
    for budget in $(arm_order "$rep"); do run_cell "$budget" "$(phase "$rep")" "$mode" "$c" "$CELL_S"; done
  done; done; done
  echo ALL_DONE
}

# A/B/A/B on one server: even repeats run the budget modes in reverse order.
arm_order() { if (( $1 % 2 )); then echo "$BUDGETS"; else tr ' ' '\n' <<< "$BUDGETS" | tac | tr '\n' ' '; fi; }

phase() { if (( REPEATS > 1 )); then echo "r$1"; else echo main; fi; }

start_server() {
  echo "$GPU_ID $(nvidia-smi --query-gpu=name,uuid --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES")" > gpu.txt
  git -C "$VLLM_DIR" log -2 --format='%H %cs %s' > commit.txt
  local serve; serve=$(serve_args)
  echo "$serve" > serve-command.txt
  ! curl -s "http://127.0.0.1:$PORT/health" >/dev/null || { echo "port $PORT busy" >&2; exit 1; }
  eval "local args=($serve)"
  "${args[@]}" > server.log 2>&1 &
  SERVER_PID=$!
  timeout 1800 bash -c "until curl -sf http://127.0.0.1:$PORT/health >/dev/null; do kill -0 $SERVER_PID || exit 1; sleep 5; done"
}

serve_args() {
  local spec="{\"model\":\"$SPEC_MODEL\",\"num_speculative_tokens\":$K,\"method\":\"$METHOD\",\"enable_adaptive_verification\":true}"
  local args=(vllm serve "$MODEL" --port "$PORT" --max-model-len 16384 --max-num-seqs "$MAX_SEQS"
    --max-num-batched-tokens 16384 --gpu-memory-utilization 0.85 --no-enable-prefix-caching
    --disable-uvicorn-access-log --speculative-config "$spec"
    --default-chat-template-kwargs '{"enable_thinking": true}'
    --compilation-config "{\"max_cudagraph_capture_size\": $((MAX_SEQS * (K + 1)))}" ${SERVE_ARGS:-})
  printf '%q ' "${args[@]}"
}

run_cell() {  # <budget mode> <phase> <sampling mode> <concurrency> <seconds>
  local budget=$1 phase=$2 mode=$3 c=$4 secs=$5 pids=() group
  echo "$budget $phase $mode c$c" > control.txt
  echo "CELL $budget $phase $mode c$c $(date -u +%FT%TZ)"
  while read -r group; do
    load "$phase-$budget-$mode-c$c" "${group%% *}" "$secs" "${group#* }" &
    pids+=($!)
  done < <(client_groups "$mode" "$c")
  wait "${pids[@]}"
}

# Prints one line per client group: <concurrency> <extra request inputs as JSON>.
client_groups() {  # <sampling mode> <concurrency>
  local c=$2 some=$(( $2 / 4 > 0 ? $2 / 4 : 1 ))
  local sampled='"temperature":0.7,"top_p":0.8,"top_k":20'
  case $1 in
    greedy)     echo "$c {\"temperature\":0}" ;;
    temp)       echo "$c {\"temperature\":0.7,\"top_p\":1,\"top_k\":-1}" ;;  # explicit: the model's generation_config sets top-k/top-p defaults
    topp)       echo "$c {\"temperature\":0.7,\"top_p\":0.8,\"top_k\":-1}" ;;
    topk_topp)  echo "$c {$sampled}" ;;
    penalties)  echo "$c {$sampled,\"presence_penalty\":1.5}" ;;
    structured) echo "$((c - some)) {\"temperature\":0}"  # a grammar bitmask on a quarter of the requests
                echo "$some {\"temperature\":0,\"response_format\":{\"type\":\"json_object\"},\"chat_template_kwargs\":{\"enable_thinking\":false}}" ;;
    mixed)      echo "$((c - 1)) {\"temperature\":0}"  # one sampling request among greedy ones
                echo "1 {$sampled}" ;;
    *) echo "unknown mode $1" >&2; exit 1 ;;
  esac
}

load() {  # <cell name> <concurrency> <seconds> <extra inputs JSON>
  local name=$1-g$2 c=$2 secs=$3 extra=$4
  [[ $c -gt 0 ]] || return 0
  HF_HUB_OFFLINE=${AIPERF_HF_OFFLINE:-1} timeout 600 "$HOME/.venv/bin/aiperf" profile --model "$MODEL" --tokenizer "$TOKENIZER" --url "http://127.0.0.1:$PORT" \
    --endpoint-type chat --streaming --input-file "$ROOT/prompts/mixed.jsonl" \
    --custom-dataset-type single_turn --dataset-sampling-strategy sequential \
    --concurrency "$c" --benchmark-duration "$secs" --benchmark-grace-period 60 \
    --extra-inputs "{\"max_tokens\":512,${extra#\{}" \
    --artifact-dir "aiperf/$name" --ui-type none > "aiperf-$name.log" 2>&1 \
    || echo "aiperf $name exit $?" >> failures.txt
}

cleanup() {
  local code=$?
  [[ -z $SERVER_PID ]] || { kill "$SERVER_PID" || true; wait "$SERVER_PID" || true; }
  echo "$code" > exit-code.txt
}

main
