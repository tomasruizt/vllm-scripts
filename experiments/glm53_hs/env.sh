export PATH=/home/tomasruizt/.venv/bin:/usr/local/cuda-13.0/bin:$PATH
export HS_RUNS_DIR=${HS_RUNS_DIR:-$HOME/benchmarks/glm-hs-validation}
export VLLM_DIR=${VLLM_DIR:-$HOME/code/vllm}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
export MODEL_DIR=${MODEL_DIR:-/data/yewentao256/glm53-flash-mtp3-vs-dspark8/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a}
export VLLM_KV_CACHE_LAYOUT=BLHNC
export TP_SIZE=4
export TOKENIZERS_PARALLELISM=false
