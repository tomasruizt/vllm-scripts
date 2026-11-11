#!/bin/bash
# RTX pod (pirate2, ns runtime) for the V131k top-p study. Usage: v131k_pod.sh up | push | run NAME ARGS... | fetch NAME | down
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
oc=(oc --kubeconfig "$HOME/.kube/pirate2" -n runtime)
pod=sm120-v131k
out=$here/../results/sm120-v131k

case $1 in
up)
    "${oc[@]}" apply -f "$here/pod_v131k.yaml"
    "${oc[@]}" wait --for=condition=Ready "pod/$pod" --timeout=600s ;;
push)
    "${oc[@]}" exec $pod -- mkdir -p /tmp/work/code
    for f in head_triton head_timed serp_triton serp_timed phases_v131k ${EXTRA:-}; do
        "${oc[@]}" cp "$here/$f.py" "$pod:/tmp/work/code/$f.py"
    done ;;
run)
    name=$2; shift 2
    cmd="cd /tmp/work/code && TRITON_CACHE_DIR=/tmp/work/triton python3 phases_v131k.py /tmp/work/$name.jsonl $* > /tmp/work/$name.log 2>&1; echo EXIT \$? >> /tmp/work/$name.log"
    "${oc[@]}" exec $pod -- bash -c "nohup bash -c '$cmd' > /dev/null 2>&1 < /dev/null &" ;;
wait)
    timeout 2400 "${oc[@]}" exec $pod -- bash -c "until grep -q EXIT /tmp/work/$2.log; do sleep 10; done; tail -n 3 /tmp/work/$2.log | cut -c1-200" ;;
fetch)
    mkdir -p "$out"
    "${oc[@]}" cp "$pod:/tmp/work/$2.jsonl" "$out/$2.jsonl" ;;
down)
    "${oc[@]}" delete pod $pod --wait=false ;;
esac
