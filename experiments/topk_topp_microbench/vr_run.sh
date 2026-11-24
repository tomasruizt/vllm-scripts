#!/usr/bin/env bash
# Benchmark main's vs. the PR's top-k/top-p Triton kernel on the Vera Rubin SLURM cluster.
# One sbatch job on one GPU, in the vLLM image: vr_ab.py runs both kernels in one process,
# alternating per case (impl order reversed every other pass), with the exact check.
# Per pass: the report suite (synthetic + real Qwen3.6 rows) and the GLM-5.3 top-p rows.
# Usage: vr_run.sh RUN_ID [BENCH_ARGS...]  (results: REMOTE:runs/RUN_ID, fetch with
#   make fetch-results REMOTE=vera-rubin-preview RUN_ID=RUN_ID)
# Env: REMOTE, LOCAL (PR worktree), IMAGE (.sqsh on the NFS home), PASSES, TIME (sbatch limit), GLM=0.
set -euo pipefail
run_id=$1
shift
remote=${REMOTE:-vera-rubin-preview}
local_tree=${LOCAL:-$HOME/code/rh/vllm-topk-hier}
here=$(cd "$(dirname "$0")" && pwd)
out=runs/$run_id

main() {
    ssh -o BatchMode=yes "$remote" "mkdir -p $out/bench_fixed data"
    scp -q "$local_tree/vllm/v1/sample/ops/topk_topp_triton.py" "$remote:$out/new_kernel.py"
    scp -q "$here/vr_ab.py" "$remote:$out/"
    scp -q "$here/bench_fixed/benchmark_topk_topp.py" "$remote:$out/bench_fixed/"
    ssh -o BatchMode=yes "$remote" "ls data/qwen36-verify-logits-t1.pt data/glm53-verify-logits.pt" > /dev/null 2>&1 ||
        scp -q "$here"/results/{qwen36-verify-logits-t1,glm53-verify-logits}.pt "$remote:data/"
    git -C "$local_tree" rev-parse HEAD | ssh -o BatchMode=yes "$remote" "cat > $out/commit.txt"
    job_script "$@" | ssh -o BatchMode=yes "$remote" "cat > $out/job.sh && cd $out && sbatch --parsable job.sh"
}

job_script() {
    cat <<EOF
#!/usr/bin/env bash
#SBATCH --partition=batch --nodes=1 --gpus-per-node=1 --time=${TIME:-01:00:00}
#SBATCH --job-name=topk-$run_id --output=slurm-%j.log
set -uo pipefail
image=\$HOME/images/${IMAGE:-vllm-cu135-8586ccbc.sqsh}
mkdir -p results /raid/users/\$USER/triton-cache
export TRITON_CACHE_DIR=/scratch/triton-cache PYTHONPATH=/scratch/py314-libs
in_image() {
    srun --ntasks=1 --container-image="\$image" --container-workdir=/work \\
        --container-mounts="\$PWD:/work,\$HOME/data:/data,/raid/users/\$USER:/scratch" "\$@"
}
# The image has no pandas (the benchmark prints its table with it): install it on the tray's /raid.
in_image bash -c "python3 -c 'import pandas' 2>/dev/null || uv pip install -q --python python3 --target /scratch/py314-libs pandas" > setup.log 2>&1
in_image nvidia-smi --query-gpu=name,compute_cap,driver_version,clocks.max.sm --format=csv > gpu.txt

bench() {  # bench NAME IMPLS LOGITS ARGS...
    local name=\$1 impls=\$2 logits=\$3
    shift 3
    in_image python3 vr_ab.py new_kernel.py --implementations \$impls --check --logits-file /data/\$logits \\
        --output results/\$name.jsonl "\$@" > \$name.log 2>&1
    echo "\$name exit=\$?"
}

for p in \$(seq 0 \$((${PASSES:-3} - 1))); do
    impls="triton new"; (( p % 2 )) && impls="new triton"
    bench report_r\$p "\$impls" qwen36-verify-logits-t1.pt --batch-sizes 8 16 64 128 256 512 1024 2048 --vocab-sizes 131072 248320 $*
    [[ ${GLM:-1} == 1 ]] && bench glm_r\$p "\$impls" glm53-verify-logits.pt --batch-sizes 8 16 64 128 256 512 1024 2048 --scenarios real_topp $*
done
echo DONE
EOF
}

main "$@"
