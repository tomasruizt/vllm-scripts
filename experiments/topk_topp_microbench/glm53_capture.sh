#!/usr/bin/env bash
# Capture pre-filter verify logits of GLM-5.3-Flash (TP4, FP8 KV, MTP5, generation_config
# sampling defaults: T 1.0, top-p 0.95) with capture_probe.py, on the B300 from main
# (~/code/topk-cuda-main). Three AIPerf chat runs at C32, one probe arm (8 captures) each.
# Async. Usage: glm53_capture.sh  (log: ~/runs/glm53-verify-logits/run.log on the host)
set -euo pipefail
remote=${REMOTE:-RH-nm-ore-b300-01}
here=$(cd "$(dirname "$0")" && pwd)
opts=(-o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)
out=runs/glm53-verify-logits

main() {
    ssh -T -n "${opts[@]}" "$remote" "mkdir -p $out/probe"
    scp -q "${opts[@]}" "$here/capture_probe.py" "$here/capture_probe_tp.py" "$here/benchmark.py" "$remote:$out/probe/"
    sed '/temperature: 0/d' "$here/../glm53_flash_capture_grid/benchmark.yaml" | ssh -T "${opts[@]}" "$remote" "cat > $out/benchmark.yaml"
    gpu_script | ssh -T "${opts[@]}" "$remote" "cat > $out/gpu.sh"
    local run="chg run --gpus 4 --timeout 45m --note glm53-verify-logits -- bash \$HOME/$out/gpu.sh"
    ssh -T -n "${opts[@]}" "$remote" "setsid nohup bash -c '$run; echo EXIT \$?' > $out/run.log 2>&1 < /dev/null & disown"
    echo "launched $remote:$out"
}

gpu_script() {
    cat <<'EOF'
set -euo pipefail
cd $HOME/runs/glm53-verify-logits
MODEL=$HOME/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a
PORT=8779 URL=http://127.0.0.1:8779 AIPERF=$HOME/aiperf-venv/bin/aiperf
export PATH=$HOME/code/topk-cuda-main/.venv/bin:$PATH PYTHONPATH=$PWD/probe VLLM_SERVER_DEV_MODE=1 HF_HUB_OFFLINE=1
unset VLLM_USE_V2_MODEL_RUNNER

rpc() { curl -sf -X POST $URL/collective_rpc -H 'Content-Type: application/json' -d "{\"method\":\"$1\",\"args\":$2,\"timeout\":120}"; echo; }
stop() { kill -TERM -- -$SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true; echo SERVER_STOPPED; }

nvidia-smi --query-gpu=index,name --format=csv,noheader
git -C $HOME/code/topk-cuda-main log -1 --format='%H %ad %s' --date=short > commit.txt
command=(python -m vllm.entrypoints.openai.api_server --model $MODEL --served-model-name glm53-flash
  --tensor-parallel-size 4 --kv-cache-dtype fp8
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47
  --worker-extension-cls capture_probe_tp.RankZeroProbe
  --disable-uvicorn-access-log --host 127.0.0.1 --port $PORT)
printf '%q ' "${command[@]}" > server-command.txt
setsid "${command[@]}" > server.log 2>&1 &
SERVER=$!
trap stop EXIT
deadline=$((SECONDS + 1500))
until curl -sf --max-time 2 $URL/health >/dev/null; do
  kill -0 $SERVER 2>/dev/null || { tail -n 20 server.log; exit 1; }
  (( SECONDS < deadline )) || { echo startup timed out; exit 1; }
  sleep 5
done
echo "READY $SECONDS s"
rpc init_worker_extension '[]'
for run in 0 1 2; do
  rpc arm_capture "[\"$PWD/captures/run$run\", 64]"
  sed "s|__PORT__|$PORT|g; s|__MODEL__|zai-org/GLM-5.3-Flash|g; s|__ARTIFACTS__|$PWD/aiperf/run$run|g; s|__CONCURRENCIES__|32|g; s|__DURATIONS__|30|g" benchmark.yaml > aiperf-run$run.yaml
  timeout 300 $AIPERF profile --config aiperf-run$run.yaml > aiperf-run$run.log 2>&1 || echo "aiperf run$run failed"
  rpc capture_status '[]' | tee capture-status-run$run.json
done
echo GPU_DONE
EOF
}

main
