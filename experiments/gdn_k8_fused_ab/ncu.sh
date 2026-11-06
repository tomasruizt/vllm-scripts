#!/usr/bin/env bash
# NCU full set of the fused GDN MTP decode kernel per label (launches after the first 3). Run from the run directory inside chg.
# Usage: ncu.sh [ncu_profile.py args...]   Labels come from LABELS (main or a gdn-k8-<label> worktree).
set -euo pipefail
CODE=${CODE:-$HOME/code}
MAIN=${MAIN:-$CODE/gdn-k8-verify-b300-main}
LABELS=${LABELS:-main splitv3}
NCU_FLAGS=${NCU_FLAGS:-}  # e.g. --clock-control none --cache-control none
export PATH=$MAIN/.venv/bin:$PATH:/usr/local/cuda/bin

for label in $LABELS; do
    dir=$CODE/gdn-k8-$label
    [[ $label != main ]] || dir=$MAIN
    PYTHONPATH=$dir ncu --set full --import-source yes $NCU_FLAGS --kernel-name regex:gdn_decode_post_conv_mtp --launch-skip 3 --launch-count ${NCU_LAUNCHES:-1} \
        -f -o "ncu-$label" python ncu_profile.py "$@" > "ncu-$label.log" 2>&1
    echo "ncu $label: exit $?"
done
echo NCU_DONE
