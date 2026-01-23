#!/bin/bash
#SBATCH --partition kisski-h100
#SBATCH --gpus H100:{N_GPUS}
#SBATCH --array=0
#SBATCH --output /user/m.ruiztellez/u20468/vllm-scripts/slurm-results/vllm/logs/%x/%a_out.logs
#SBATCH --error /user/m.ruiztellez/u20468/vllm-scripts/slurm-results/vllm/logs/%x/%a_err.logs
#SBATCH --job-name {JOB_NAME}
#SBATCH --time 01:00:00

# Calculate port based on base port and array task ID to avoid conflicts
export VLLM_PORT=$((15000 + $SLURM_ARRAY_TASK_ID))
export HF_TOKEN=$HF_TOKEN
export HF_HOME=$HF_HOME
export TMPDIR=$LOCAL_TMPDIR
export VLLM_USE_V1=1


source /user/m.ruiztellez/u21521/venv/bin/activate

# Launch vLLM server in background
echo 'Starting vLLM server...'
vllm serve {TARGET_MODEL} \
  --tensor-parallel-size {N_GPUS} \
  --no-enable-prefix-caching \
  --max-model-len 5000 \
  --disable-uvicorn-access-log \
  --port $VLLM_PORT &

SERVER_PID=\$!
echo \"Server started with PID: \$SERVER_PID\"

# Run benchmark (has built-in wait mechanism)
for MAX_CONCURRENCY in {CONCURRENCIES}; do

  NUM_PROMPTS={NUM_PROMPTS}
  {NUM_PROMPTS_EXTRA}

  echo \"Starting benchmark with MAX_CONCURRENCY = \$MAX_CONCURRENCY and NUM_PROMPTS = \$NUM_PROMPTS...\"
  vllm bench serve \
    --model {TARGET_MODEL} \
    --dataset-name hf \
    --dataset-path {DATASET} \
    --num-prompts \$NUM_PROMPTS \
    --max-concurrency \$MAX_CONCURRENCY \
    --request-rate \$MAX_CONCURRENCY \
    --temperature {TEMPERATURE} \
    --top-p 1.0 \
    --port $VLLM_PORT \
    --ready-check-timeout-sec 600
done

echo 'Benchmark completed, stopping server...'
kill \$SERVER_PID