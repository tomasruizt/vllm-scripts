#!/usr/bin/env bash
# One AgentX point traced by the NVIDIA Nsight Perf SDK for AI (beta preview update 1). Single GPU, all in one container.
# The daemon traces the vLLM server from the outside, and the continuous exporter writes the trace to DuckDB and parquet.
# Requirements found on a B200 node with kernel driver 595.71.05:
#   * the server loads the CUDA 13.5 user-mode driver (compat package), which is what emits the CUDA API trace,
#   * the daemon needs CAP_PERFMON for GPU kernel timings and PM counters (see nsight.yaml). Without it, only the
#     CPU side (API calls, launches) and memcpy timings are recorded.
# Model: the small GLM-5.3-Flash clone (no MTP weights, so no spec decode), served under the real model's name for bench.sh.
# Env: RUN_ID, CONC, DURATION, WARMUP. prepare.sh puts the SDK and the compat package under /models/nsight, and the model,
# the AgentX client and the traces come from ../glm53_agentx (same model disk).
set -uo pipefail

export OUT=/models/results/$RUN_ID
SDK=/models/nsight/nsight-perf-sdk-for-ai-beta-preview-linux-x86_64/bin
COMPAT=/models/nsight/compat135/usr/local/cuda-13.5/compat
export XDG_RUNTIME_DIR=/tmp/xdg VLLM_USE_RUST_FRONTEND=1
mkdir -p "$OUT" "$XDG_RUNTIME_DIR"
cp /scripts/*.sh "$OUT/"
nvidia-smi --query-gpu=name,driver_version --format=csv > "$OUT/gpu.txt"

"$SDK/nvidia-nsight-perf" > "$OUT/daemon.log" 2>&1 &
"$SDK/nvtrace_poll_until_service_ready" > "$OUT/daemon-ready.log" 2>&1

LD_LIBRARY_PATH=$COMPAT vllm serve inference-optimization/GLM-5.3-Flash-2.6B-A2.3B-small \
    --served-model-name zai-org/GLM-5.3-Flash \
    --tokenizer-mode glm5_next \
    --kv-cache-dtype fp8 \
    --language-model-only \
    --tool-call-parser glm47 \
    --reasoning-parser glm47 \
    --enable-auto-tool-choice \
    --disable-uvicorn-access-log \
    --port 8000 > "$OUT/server.log" 2>&1 &
server=$!
until curl -sf http://127.0.0.1:8000/health > /dev/null; do
    kill -0 $server 2> /dev/null || { echo "SERVER_DIED" | tee "$OUT/status.txt"; exit 1; }
    sleep 5
done
echo "server healthy after ${SECONDS}s" | tee "$OUT/startup.txt"

# The exporter covers the client run plus its setup. It stops by itself and then writes parquet_export/.
export_ms=$(((DURATION + 120) * 1000))
"$SDK/nvtrace_continuous_export" --output-dir "$OUT/nvtrace" --snapshot-ms 30000 --duration-ms $export_ms \
    --stop-on-enter-key false > "$OUT/export.log" 2>&1 &
exporter=$!

bash /scripts/bench.sh > "$OUT/client.log" 2>&1
client_rc=$?
wait $exporter
kill $server
wait $server
echo "DONE client_rc=$client_rc elapsed_time=${SECONDS}s" | tee "$OUT/status.txt"
