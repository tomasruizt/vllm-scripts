#!/usr/bin/env bash
# Usage: run_one.sh <name> <port> <model> <spec-config-json> <sync|async> [--expect-av]
# Runs inside `chg run`, so CUDA_VISIBLE_DEVICES is set. Env (e.g.
# VLLM_USE_RUST_FRONTEND) is inherited from the caller.
set -uo pipefail
name=$1 port=$2 model=$3 spec=$4 mode=$5 extra=${6:-}
here=$(cd "$(dirname "$0")" && pwd)
out=$here/results/$name
mkdir -p "$out"
vllm serve "$model" --port "$port" --speculative-config "$spec" \
  --per-request-spec-decode-metrics detailed --no-enable-prefix-caching \
  --disable-uvicorn-access-log --"$( [ "$mode" = sync ] && echo no- )"async-scheduling \
  --max-model-len 4096 > "$out/server.log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
timeout 900 bash -c "until curl -sf localhost:$port/health >/dev/null; do kill -0 $pid || exit 1; sleep 2; done" \
  || { echo "SERVER_FAILED $name"; exit 1; }
python "$here/check.py" --port "$port" --name "$name" $extra > "$out/check.json" 2> "$out/check.err"
rc=$?
echo "CHECK_EXIT=$rc $name"
exit $rc
