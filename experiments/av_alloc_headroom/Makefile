SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
REMOTE ?= RH-nm-ohio-b200-01
RUN_ID ?= av-alloc-headroom-20261120
REMOTE_ROOT ?= runs
DATASETS ?= HumanEval math_reasoning writing
CONCURRENCIES ?= 16 32 64 128
MODEL ?= Qwen/Qwen3-4B
SPEC_MODEL ?= RedHatAI/Qwen3-4B-speculator.dflash2
K ?= 8
THINKING ?= true
TEMPERATURE ?= 0
TP ?= 1
CHUNK_BYTES ?= 1073741824
# mode:gpu pairs (gpu may be a comma list for TP > 1). GPUs 0, 1, 5 had CUDA context failures on 2026-11-19 (0 and 1 work again on 2026-11-20).
MODES ?= verify_all:2 policy:3
SSH := ssh -T -o BatchMode=yes -o ConnectTimeout=10
RUN := $(REMOTE_ROOT)/$(RUN_ID)
ENV := MODEL=$(MODEL) SPEC_MODEL=$(SPEC_MODEL) K=$(K) THINKING=$(THINKING) TEMPERATURE=$(TEMPERATURE) TP=$(TP) AV_TRACE_MAX_CHUNK_BYTES=$(CHUNK_BYTES) DATASETS="$(DATASETS)" CONCURRENCIES="$(CONCURRENCIES)"

.PHONY: launch watch fetch-results analyze
launch:
	$(SSH) $(REMOTE) 'mkdir -p "$(REMOTE_ROOT)" && mkdir "$(RUN)"'  # fails if the run exists: pick a new RUN_ID
	scp -q launch.sh make_prompts.py $(REMOTE):"$(RUN)/"
	$(SSH) $(REMOTE) 'cd "$(RUN)" && HF_HOME=/data/tomasruizt/huggingface ~/.venv/bin/python make_prompts.py prompts $(DATASETS)'
	$(SSH) $(REMOTE) 'cd "$(RUN)" && for m in $(MODES); do ROOT=$$PWD MODE=$${m%:*} GPU_ID=$${m#*:} $(ENV) nohup bash launch.sh > launch-$${m%:*}.log 2>&1 < /dev/null & done; echo launched'

watch:
	$(SSH) $(REMOTE) 'cd "$(RUN)"; timeout 10800 bash -c "until test -f $(word 1,$(subst :, ,$(word 1,$(MODES))))/exit-code.txt -a -f $(word 1,$(subst :, ,$(lastword $(MODES))))/exit-code.txt; do sleep 10; done"'

fetch-results:
	mkdir -p results
	$(SSH) $(REMOTE) 'tar czf - -C "$(REMOTE_ROOT)" --exclude=prompts "$(RUN_ID)"' | tar xzf - -C results

analyze:
	~/.venv/bin/python replay.py results/$(RUN_ID)
