#!/usr/bin/env bash
# Kernel tests, then the E2E A/B (main vs. the PR) of Qwen3.6 + DSpark K=7 on one Vera Rubin GPU,
# as one sbatch job in the vLLM image. Builds run interleaved on the same GPU (main, hier, then hier,
# main, for ROUNDS rounds), each a run.sh session (one server, every repeat x concurrency cell).
# hier-* mounts the PR's vllm files (from the local worktree) over the image's. Async: returns after sbatch.
# Usage: vr_e2e.sh RUN_ID  (fetch: make fetch-results REMOTE=vera-rubin-preview RUN_ID=RUN_ID)
# Env: REMOTE, LOCAL (PR worktree), IMAGE, ROUNDS, REPEATS, CONCURRENCIES, TIME, TESTS=0.
set -euo pipefail
run_id=$1
remote=${REMOTE:-vera-rubin-preview}
local_tree=${LOCAL:-$HOME/code/rh/vllm-topk-hier}
here=$(cd "$(dirname "$0")" && pwd)
out=runs/$run_id
base=fd2b7ac40f  # the PR's merge base (main on 2026-11-19), same sampler files as the image

main() {
    local files
    # nemotron_dllm.py also changed in the image after the base: keep the image's (not used here).
    files=$(git -C "$local_tree" diff --name-only "$base" HEAD -- vllm | grep -v nemotron_dllm)
    ssh -o BatchMode=yes "$remote" "mkdir -p $out/overlay"
    (cd "$local_tree" && COPYFILE_DISABLE=1 tar --no-xattrs -czf - $files tests) | ssh -o BatchMode=yes "$remote" "tar -xzf - -C $out/overlay"
    echo "$files" | ssh -o BatchMode=yes "$remote" "cat > $out/overlay/files.txt"
    scp -q "$here"/{run.sh,benchmark-default.yaml} "$here/../../benchmarking/host_load.sh" "$remote:$out/"
    git -C "$local_tree" rev-parse HEAD | ssh -o BatchMode=yes "$remote" "cat > $out/commit.txt"
    job_script | ssh -o BatchMode=yes "$remote" "cat > $out/job.sh && cd $out && sbatch --parsable job.sh"
}

job_script() {
    cat <<EOF
#!/usr/bin/env bash
#SBATCH --partition=batch --nodes=1 --gpus-per-node=1 --time=${TIME:-04:00:00}
#SBATCH --job-name=topk-e2e-$run_id --output=slurm-%j.log
set -uo pipefail
image=\$HOME/images/${IMAGE:-vllm-cu135-8586ccbc.sqsh}
site=/usr/local/lib/python3.14/dist-packages
scratch=/raid/users/\$USER
cp \$HOME/data/throughput_1k.jsonl .
mkdir -p \$scratch/home \$scratch/hub py/bin && ln -sf /usr/bin/python3 py/bin/python
export HOME=/scratch/home HF_HUB_CACHE=/scratch/hub TRITON_CACHE_DIR=/scratch/triton-cache
in_image() {  # in_image MOUNTS CMD...
    local mounts="\$PWD:/work,\$scratch:/scratch\${1:+,\$1}"
    shift
    srun --ntasks=1 --container-image="\$image" --container-mounts="\$mounts" --container-workdir=/work "\$@"
}
pr_mounts=\$(sed "s|.*|\$PWD/overlay/&:\$site/&|" overlay/files.txt | paste -sd,)

# AIPerf (Python 3.12 venv, as on B300) and the weights, on the tray's /raid.
in_image "" bash -c 'export UV_PYTHON_INSTALL_DIR=/scratch/uv-python
    [ -x /scratch/aiperf-venv/bin/aiperf ] || { rm -rf /scratch/aiperf-venv && uv venv -q --managed-python -p 3.12 /scratch/aiperf-venv && uv pip install -q -p /scratch/aiperf-venv aiperf==0.13.0; }
    hf download Qwen/Qwen3.6-35B-A3B --exclude "*.pth" > /dev/null && hf download RedHatAI/Qwen3.6-35B-A3B-speculator.dspark > /dev/null
    /scratch/aiperf-venv/bin/aiperf --version; command -v curl setsid timeout' > setup.log 2>&1
echo "setup exit=\$?"

if [[ ${TESTS:-1} == 1 ]]; then
    in_image "\$pr_mounts,\$PWD/overlay/tests:/work/tests" bash -c 'export PYTHONPATH=/scratch/py314-libs
        python3 -c "import tblib" 2>/dev/null || uv pip install -q --python python3 --target /scratch/py314-libs --no-deps tblib
        python3 -m pytest tests/v1/sample/test_topk_topp_sampler.py -q -p no:cacheprovider -k TritonTopkTopp' > pytest.log 2>&1
    echo "pytest exit=\$?"; tail -n 1 pytest.log
fi

session() {  # session BUILD ROUND
    local variant=\$1-k7-r\$2 mounts=""
    [[ \$1 == hier ]] && mounts=\$pr_mounts
    echo "SESSION \$variant \$(date -u +%FT%TZ)"
    in_image "\$mounts" env PRESET=qwen-default VENV_OVERRIDE=/work/py SRC_OVERRIDE= MAIN=/nonexistent IMAGE=\$image \\
        AIPERF=/scratch/aiperf-venv/bin/aiperf DATASET=/work/throughput_1k.jsonl HUB_CACHE=/scratch/hub \\
        MAX_MODEL_LEN=40960 GSM8K_QUESTIONS=0 REPEATS="${REPEATS:-1 2 3}" CONCURRENCIES="${CONCURRENCIES:-1 8 32 64 128 256}" \\
        bash run.sh \$variant > \$variant.log 2>&1
    echo "\$variant exit=\$?"
}

for round in \$(seq ${ROUNDS:-2}); do  # main, hier, then hier, main
    order="main hier"; (( round % 2 )) || order="hier main"
    for build in \$order; do session \$build \$round; done
done
echo DONE
EOF
}

main "$@"
