#!/usr/bin/env bash
# Per-step timing of scoring variants. Each ";"-separated sequence runs its
# variants one after another on one GPU (under `chg run`), so A/B pairs share
# hardware: e.g. SEQUENCES="ref,fused;fused,ref".
set -euo pipefail
export ROOT=${ROOT:?}
cd "$ROOT"
trap 'printf "%s\n" "$?" > launch-exit-code.txt' EXIT
export CUDA_HOME=/usr/local/cuda-13.0
TREE=${TREE:-$HOME/code/vllm-dspine}
export PATH="$TREE/.venv/bin:$CUDA_HOME/bin:$PATH"
export HF_HOME=${HF_HOME_DIR:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1
unset VLLM_USE_V2_MODEL_RUNNER
SEQUENCES=${SEQUENCES:-ref,fused;fused,ref}
export PROFILE_ARGS=${PROFILE_ARGS:-}

run_sequence() {
    local slot=$1 variants=$2
    # A variant is <scoring>[_tf32][-k<num_spec_tokens>], e.g. fused-k7.
    for variant in ${variants//,/ }; do
        local base=${variant%%-k*} k=15
        [[ $variant == *-k* ]] && k=${variant##*-k}
        local scoring=${base%_tf32} tf32=0
        [[ $base == *_tf32 ]] && tf32=1
        VLLM_DSPINE_SCORING=$scoring VLLM_DSPINE_SCORING_TF32=$tf32 \
            python profile_steps.py --out "steps-$variant-s$slot.jsonl" $PROFILE_ARGS \
            --num-spec-tokens "$k" > "$variant-s$slot.log" 2>&1
    done
}
export -f run_sequence

PIDS=()
slot=0
IFS=';' read -ra SEQS <<< "$SEQUENCES"
for seq in "${SEQS[@]}"; do
    chg run --gpus 1 --wait 1h --timeout 2h --note "DSpine scoring E2E profile: $seq" -- \
        bash -c "nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader; run_sequence $slot $seq" \
        > "sequence-s$slot.log" 2>&1 &
    PIDS+=("$!")
    slot=$((slot + 1))
done
RESULT=0
for pid in "${PIDS[@]}"; do wait "$pid" || RESULT=1; done
exit "$RESULT"
