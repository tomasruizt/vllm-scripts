#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres=gpu:{N_GPUS}
#SBATCH --array=0
#SBATCH -o /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/vllm/logs/%x/%a_out.logs
#SBATCH -e /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/vllm/logs/%x/%a_err.logs
#SBATCH --job-name {JOB_NAME}
#SBATCH --time 00:30:00

# Set container name based on job name and array task ID to avoid conflicts
CONTAINER_NAME={JOB_NAME}

# Calculate port based on base port and array task ID to avoid conflicts
VLLM_PORT={VLLM_PORT}

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
    $CONTAINER_NAME bash -c "
        source /root/ptvenv/bin/activate
        
        # Launch vLLM server in background
        echo 'Starting vLLM server...'
        vllm serve Qwen/Qwen3-32B \
          --tensor-parallel-size {N_GPUS} \
          --no-enable-prefix-caching \
          --speculative_config.method={SPECULATIVE_METHOD} \
          --speculative_config.model={SPECULATIVE_MODEL} \
          --speculative_config.num_speculative_tokens={N_SPEC_TOKS} \
          --speculative_config.max_model_len=20000 \
          --max-model-len 20000 \
          --disable-uvicorn-access-log \
          --port $VLLM_PORT &
        
        SERVER_PID=\$!
        echo \"Server started with PID: \$SERVER_PID\"
        
        # Run benchmark (has built-in wait mechanism)
        for MAX_CONCURRENCY in {CONCURRENCIES}; do

          NUM_PROMPTS={NUM_PROMPTS}

          echo \"Starting benchmark with MAX_CONCURRENCY = \$MAX_CONCURRENCY and NUM_PROMPTS = \$NUM_PROMPTS...\"
          vllm bench serve \
            --model Qwen/Qwen3-32B \
            --dataset-name hf \
            --dataset-path {DATASET} \
            --num-prompts \$NUM_PROMPTS \
            --max-concurrency \$MAX_CONCURRENCY \
            --temperature {TEMPERATURE} \
            --top-p 1.0 \
            --port $VLLM_PORT \
            --ready-check-timeout-sec 600
        done
        
        echo 'Benchmark completed, stopping server...'
        kill \$SERVER_PID
    "

# Clean up: remove the container after job completion
echo "Cleaning up: removing container $CONTAINER_NAME..."
enroot remove --force $CONTAINER_NAME || true