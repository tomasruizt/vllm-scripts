#!/usr/bin/env bash
# One server per MODE (verify_all | policy), sweeping datasets x concurrencies with AIPerf.
# The AV trace (per request, per step) goes to $ROOT/$MODE/trace.jsonl, and
# manifest.jsonl records the wall-clock window of each workload.
set -euo pipefail
ROOT=${ROOT:?} MODE=${MODE:?} GPU_ID=${GPU_ID:?}
MODEL=${MODEL:-Qwen/Qwen3-4B}
SPEC_MODEL=${SPEC_MODEL:-RedHatAI/Qwen3-4B-speculator.dflash2}
K=${K:-8}
THINKING=${THINKING:-true}
TEMPERATURE=${TEMPERATURE:-0}
TP=${TP:-1}
DATASETS=${DATASETS:-HumanEval math_reasoning writing}
CONCURRENCIES=${CONCURRENCIES:-16 32 64 128}
PORT=$((8700 + ${GPU_ID%%,*}))
OUT=$ROOT/$MODE
mkdir -p "$OUT"
cd "$OUT"
[[ ${1:-} == inner ]] || trap 'printf "%s\n" "$?" > exit-code.txt' EXIT

export HF_HOME=/data/tomasruizt/huggingface HF_HUB_OFFLINE=1
export CUDA_HOME=/usr/local/cuda-13.0
VENV=$HOME/code/vllm-av-headroom/.venv
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export AV_TRACE_PATH=$OUT/trace.jsonl
# Lifts the 1GB rejection-sampler chunk cap (~1,766 logits at a 152K vocab), which also caps
# the AV budget, so verify-all stays uncensored at high concurrency.
export AV_TRACE_MAX_CHUNK_BYTES=${AV_TRACE_MAX_CHUNK_BYTES:-$((1 << 30))}
export AV_TRACE_VERIFY_ALL=$([[ $MODE == verify_all ]] && echo 1 || echo 0)
git -C "$HOME/code/vllm-av-headroom" log -1 --format='%H %s' > vllm-commit.txt

SPEC="{\"model\":\"$SPEC_MODEL\",\"num_speculative_tokens\":$K,\"method\":\"dflash\",\"enable_adaptive_verification\":true}"
MAX_NUM_SEQS=$(tr ' ' '\n' <<< "$CONCURRENCIES" | sort -n | tail -1)
SERVE=(vllm serve "$MODEL" --port "$PORT" --max-model-len 16384 --max-num-seqs "$MAX_NUM_SEQS" --tensor-parallel-size "$TP"
  --max-num-batched-tokens 16384 --gpu-memory-utilization 0.85 --no-enable-prefix-caching
  --disable-uvicorn-access-log --speculative-config "$SPEC"
  --default-chat-template-kwargs "{\"enable_thinking\": $THINKING}"
  # Capture graphs up to the largest verify-all decode step, so no tested regime falls off
  # the graphs (the default cap is 1,024 tokens on Blackwell).
  --compilation-config "{\"max_cudagraph_capture_size\": $((MAX_NUM_SEQS * (K + 1)))}")
printf '%q ' "${SERVE[@]}" > serve-command.txt

run_workload() {  # <dataset> <concurrency>
  local ds=$1 c=$2 n=$(( $2 * 4 > 128 ? $2 * 4 : 128 ))
  local t0; t0=$(date +%s.%N)
  "$HOME/.venv/bin/aiperf" profile --model "$MODEL" --tokenizer "$MODEL" --url "http://127.0.0.1:$PORT" \
    --endpoint-type chat --streaming --input-file "$ROOT/prompts/$ds.jsonl" \
    --custom-dataset-type single_turn --dataset-sampling-strategy sequential \
    --concurrency "$c" --request-count "$n" \
    --extra-inputs max_tokens:1024 --extra-inputs temperature:$TEMPERATURE \
    --artifact-dir "aiperf/$ds-c$c" --ui-type none > "aiperf-$ds-c$c.log" 2>&1 \
    || echo "aiperf $ds c$c exit $?" >> failures.txt
  printf '{"dataset":"%s","concurrency":%d,"requests":%d,"t0":%s,"t1":%s}\n' \
    "$ds" "$c" "$n" "$t0" "$(date +%s.%N)" >> manifest.jsonl
}

inner() {
  "${SERVE[@]}" > server.log 2>&1 &
  local pid=$!
  timeout 1800 bash -c "until curl -sf http://127.0.0.1:$PORT/health >/dev/null; do sleep 5; done"
  for ds in $DATASETS; do for c in $CONCURRENCIES; do run_workload "$ds" "$c"; done; done
  kill $pid; wait $pid || true
}

if [[ ${1:-} == inner ]]; then inner; exit; fi
chg run --gpu-ids "$GPU_ID" --timeout 3h --note "AV allocation headroom trace ($MODE)" -- \
  bash "$ROOT/launch.sh" inner
