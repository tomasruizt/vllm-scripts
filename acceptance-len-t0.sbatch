#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --array=0
#SBATCH -o /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/vllm/logs/%x/%a_out.logs
#SBATCH -e /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/vllm/logs/%x/%a_err.logs
#SBATCH --job-name vllm-acceptance-len-t0
#SBATCH --time 1:00:00

# Set container name based on job name and array task ID to avoid conflicts
CONTAINER_NAME="vllm-acceptance-len-t0_${SLURM_ARRAY_TASK_ID}"

# Calculate port based on base port and array task ID to avoid conflicts
VLLM_PORT=$((15000 + $SLURM_ARRAY_TASK_ID))

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
    $CONTAINER_NAME bash -c "source /root/ptvenv/bin/activate && python3 /workspace/code/vllm/examples/offline_inference/spec_decode.py \
        --model-dir Qwen/Qwen3-32B \
        --draft-model Qwen/Qwen3-1.7B \
        --method draft_model \
        --num_spec_tokens 3 \
        --dataset-name hf \
        --dataset-path philschmid/mt-bench \
        --num_prompts 100 \
        --temp 0.0 \
        --gpu-memory-utilization 0.9"

# Clean up: remove the container after job completion
echo "Cleaning up: removing container $CONTAINER_NAME..."
enroot remove --force $CONTAINER_NAME || true