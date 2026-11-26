#!/usr/bin/env bash
# One AgentX point in the GPU pod: start the server, wait until healthy, run the client, stop the server.
# The server is restarted for every point, as AgentX requires. Env: RUN_ID, CONC, DURATION, WARMUP.
set -uo pipefail

export OUT=/models/results/$RUN_ID
mkdir -p "$OUT"
cp /scripts/*.sh "$OUT/"
nvidia-smi --query-gpu=index,name,uuid --format=csv > "$OUT/gpu.txt"
vllm --version > "$OUT/vllm-version.txt" 2>&1
bash /scripts/host_load.sh "$OUT/host_load.txt" &
host_load=$!

bash /scripts/serve.sh > "$OUT/server.log" 2>&1 &
server=$!
until curl -sf http://127.0.0.1:8000/health > /dev/null; do
    kill -0 $server 2> /dev/null || { echo "SERVER_DIED" | tee "$OUT/status.txt"; exit 1; }
    sleep 5
done
echo "server healthy after ${SECONDS}s" | tee "$OUT/startup.txt"

bash /scripts/bench.sh > "$OUT/client.log" 2>&1
echo "DONE client_rc=$?" | tee "$OUT/status.txt"
kill $server $host_load
wait $server
