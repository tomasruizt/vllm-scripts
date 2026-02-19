#!/bin/bash
set -euo pipefail

export VLLM_USE_V1=1

VLLM="/home/tomasruiz/code/vllm/venv/bin/python -m vllm.entrypoints.cli.main"

PROFILE_DIR="./vllm_profile-4b-sd-0.6b"
mkdir -p "$PROFILE_DIR"

MODEL="Qwen/Qwen3-4B"
DRAFT_MODEL="Qwen/Qwen3-0.6B"

# Start the server in the background
$VLLM serve "$MODEL" \
    --no-enable-prefix-caching \
    --speculative_config.method=draft_model \
    --speculative_config.model="$DRAFT_MODEL" \
    --speculative_config.num_speculative_tokens=3 \
    --speculative_config.max_model_len=2048 \
    --max-model-len 2048 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.9 \
    --profiler_config.profiler=torch \
    --profiler_config.torch_profiler_dir="$PROFILE_DIR" \
    --disable-uvicorn-access-log &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to be ready..."
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 2
done
echo "Server is ready."

# Run a small profiling workload
$VLLM bench serve \
    --model "$MODEL" \
    --dataset-name hf \
    --dataset-path likaixin/InstructCoder \
    --num-prompts 5 \
    --output-len 10 \
    --temperature 0.0 \
    --top-p 1.0 \
    --profile

echo "Profiling done. Stopping server..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true
echo "Traces saved to $PROFILE_DIR"
