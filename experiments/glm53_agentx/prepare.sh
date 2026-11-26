#!/usr/bin/env bash
# Runs once in the CPU pod: weights, AgentX client venv and traces onto the model disk, so GPU pods start right away.
# The client gets its own Python on the disk (agentx-harness needs 3.11 to 3.13, vLLM images moved to 3.14).
set -euo pipefail

REVISION=46a050bec13002ad29ee795c16e61cf4fa3c4a9c
hf download zai-org/GLM-5.3-Flash --revision $REVISION > /dev/null
# The AgentX client loads the tokenizer by repo name, which resolves through refs/main. A download by revision does not write it.
refs=/models/hf/hub/models--zai-org--GLM-5.3-Flash/refs
mkdir -p $refs && printf %s $REVISION > $refs/main
# The small clone, for smoke tests of the scripts on one GPU.
hf download inference-optimization/GLM-5.3-Flash-2.6B-A2.3B-small > /dev/null

export UV_PYTHON_INSTALL_DIR=/models/uv-python UV_CACHE_DIR=/models/uv-cache
uv venv --clear --python 3.12 --managed-python /models/agentx-venv
uv pip install -q --python /models/agentx-venv/bin/python 'agentx-harness==1.0.14' 'datasets>=4.7.0'
hf download --repo-type dataset semianalysisai/cc-traces-weka-062126 > /dev/null
/models/agentx-venv/bin/agentx --version
echo "PREPARE_DONE elapsed_time=${SECONDS}s"
