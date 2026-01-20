#!/bin/bash
#SBATCH --partition kisski
#SBATCH --gpus H100:1
#SBATCH --array=0
#SBATCH --output /projects/extern/kisski/kisski_trainings/tr_20251001-performance-ai-hpc/dir.project/vllm/%x/%a_out.logs
#SBATCH --error /projects/extern/kisski/kisski_trainings/tr_20251001-performance-ai-hpc/dir.project/vllm/%x/%a_err.logs
#SBATCH --job-name vllm-acceptance-len-t0
#SBATCH --time 00:30:00

# Calculate port based on base port and array task ID to avoid conflicts
export VLLM_PORT=$((15000 + $SLURM_ARRAY_TASK_ID))
export HF_TOKEN=$HF_TOKEN
export HF_HOME=$HF_HOME
export TMPDIR=$LOCAL_TMPDIR
export VLLM_USE_V1=1

source /user/m.ruiztellez/u21521/venv/bin/activate
python3 vllm/examples/offline_inference/spec_decode.py \
    --model-dir Qwen/Qwen3-32B \
    --draft-model Qwen/Qwen3-1.7B \
    --method draft_model \
    --num_spec_tokens 3 \
    --dataset-name hf \
    --dataset-path philschmid/mt-bench \
    --num_prompts 100 \
    --temp 0.0 \
    --gpu-memory-utilization 0.9
