#!/usr/bin/env bash
set -euo pipefail

revision=$1
model_size=$2
gpu_mode=${3:-queue}
root=/home/USER/code
runner="$root/vllm-49730/benchmarks/dflash_4b/run.py"
results="$root/vllm-49730/benchmarks/dflash_4b/results/pr57962-sweep"
engine=vllm
concurrencies=(1 8 16 32)

case "$revision" in
  main)
    checkout="$root/vllm-49730-main"
    export PYTHONPATH="$checkout:/tmp/vllm-49730-main-overlay"
    port_base=8300
    ;;
  pr)
    checkout="$root/vllm-gdn-reuse"
    export PYTHONPATH="$checkout:/tmp/vllm-gdn-reuse-overlay:/tmp/vllm-49730-main-overlay"
    port_base=8400
    ;;
  pr2)
    checkout="$root/vllm-pr52297"
    export PYTHONPATH="$checkout:/tmp/vllm-pr52297-overlay:/tmp/vllm-49730-main-overlay"
    port_base=8600
    concurrencies=(1 8)
    if [[ "$model_size" == 27B ]]; then
      concurrencies=(1)
    fi
    ;;
  sglang)
    engine=sglang
    checkout="$root/vllm-49730"
    export PYTHONPATH="$checkout/.deps/sglang_deps"
    port_base=8500
    concurrencies=(8 16)
    if [[ "$model_size" == 27B ]]; then
      concurrencies=(1 8 16)
    fi
    ;;
  *) exit 2 ;;
esac
if [[ "$model_size" == 27B ]]; then
  port_base=$((port_base + 50))
fi
export PATH="$HOME/.venv/bin:$PATH"
cd "$checkout"
git rev-parse HEAD

for concurrency in "${concurrencies[@]}"; do
  requests=$((20 * concurrency))
  warmups=$((2 * concurrency))
  if [[ "$concurrency" == 1 ]]; then
    requests=100
    warmups=10
  fi
  output="$results/$model_size/$revision/c$concurrency"
  if [[ -f "$output/${engine}_dflash/summary.json" ]]; then
    continue
  fi
  command=("$HOME/.venv/bin/python" "$runner" "$engine" dflash
    --model-size "$model_size" --concurrency "$concurrency"
    --request-count "$requests" --warmup-request-count "$warmups"
    --port "$((port_base + concurrency))" --output "$output")
  if [[ "$gpu_mode" == reserved ]]; then
    CUDA_VISIBLE_DEVICES=2 "${command[@]}"
  else
    canhazgpu run --gpus 1 --note "PR57962 $revision $model_size DFlash c$concurrency" -- "${command[@]}"
  fi
done
