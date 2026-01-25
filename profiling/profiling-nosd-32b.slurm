#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --array=0
#SBATCH -o /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/vllm/logs/%x/%a_out.logs
#SBATCH -e /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/vllm/logs/%x/%a_err.logs
#SBATCH --job-name vllm-profiling-nosd-32b
#SBATCH --time 00:30:00

# Set container name based on job name and array task ID to avoid conflicts
CONTAINER_NAME="vllm-profiling-nosd-32b"

# Delete any existing container with this name
echo "Removing any existing container named $CONTAINER_NAME..."
enroot remove --force $CONTAINER_NAME || true

# Create new container
echo "Creating new container $CONTAINER_NAME..."
enroot create --name $CONTAINER_NAME /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/tomasruiz/my_custom_pt.sqsh

# Start the container with necessary mounts and environment variables
enroot start --rw --root \
    --mount /dss/dsshome1/0D/di38bec/code:/workspace/code \
    --mount /dss/dsshome1/0D/di38bec/datasets:/workspace/datasets \
    --mount "$DSS_HOME:$DSS_HOME" \
    --env DSS_HOME=$DSS_HOME \
    --env HF_TOKEN=$HF_TOKEN \
    --env HF_HOME=$HF_HOME \
    --env VLLM_USE_V1=1 \
    --env VLLM_TORCH_PROFILER_DIR=/dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/vllm/logs/vllm-profiling-nosd-32b/profiles/ \
    --env CUDA_LAUNCH_BLOCKING=1 \
    $CONTAINER_NAME bash -c "source /root/ptvenv/bin/activate && vllm bench throughput \
        --model=Qwen/Qwen3-32B \
        --dataset-name=hf \
        --dataset-path=likaixin/InstructCoder \
        --max-num-seqs=100 \
        --num-prompts=10 \
        --input-len=1000 \
        --output-len=10 \
        --max-model-len=2048 \
        --gpu-memory-utilization=0.95 \
        --profile"

# Clean up: remove the container after job completion
echo "Cleaning up: removing container $CONTAINER_NAME..."
enroot remove --force $CONTAINER_NAME || true