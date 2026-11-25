SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
REMOTE ?= RH-nm-frk-h100-01-preserve
RUN_ID ?= av-sync-ring-20261125
REMOTE_ROOT ?= runs
VARIANTS ?= k5 av_main av_depth3 av_ring3 av_lazy
WORKLOADS ?= prefill decode
MODEL_KEY ?= 4b
GPUS_PER_CHAIN ?= 1
MAX_NUM_SEQS ?= 16
PORT_BASE ?= 8860
ENV := VARIANTS="$(VARIANTS)" WORKLOADS="$(WORKLOADS)" MODEL_KEY=$(MODEL_KEY) GPUS_PER_CHAIN=$(GPUS_PER_CHAIN) MAX_NUM_SEQS=$(MAX_NUM_SEQS) PORT_BASE=$(PORT_BASE)
SSH := ssh -T -o BatchMode=yes -o ConnectTimeout=10
RUN := $(REMOTE_ROOT)/$(RUN_ID)

.PHONY: launch watch fetch-results analyze
launch:
	$(SSH) $(REMOTE) 'mkdir -p "$(REMOTE_ROOT)" && mkdir "$(RUN)"'  # fails if the run exists: pick a new RUN_ID
	scp -q run.py launch.sh ../av_alloc_headroom/make_prompts.py $(REMOTE):"$(RUN)/"
	$(SSH) $(REMOTE) 'cd "$(RUN)" && ~/code/av-sync/main/.venv/bin/python make_prompts.py prompts mixed'
	$(SSH) $(REMOTE) 'cd "$(RUN)" && (ROOT=$$PWD $(ENV) nohup bash launch.sh > launch.log 2>&1 < /dev/null &); echo launched'

watch:
	$(SSH) $(REMOTE) 'cd "$(RUN)"; timeout 5400 bash -c "until test -f launch-exit-code.txt; do sleep 5; done"; cat launch-exit-code.txt'

fetch-results:
	mkdir -p results
	$(SSH) $(REMOTE) 'tar czf - --exclude="inputs.json" --exclude="*_raw.jsonl" --exclude="*.jsonl" --exclude="prompts" -C "$(REMOTE_ROOT)" "$(RUN_ID)"' | tar xzf - -C results

analyze:
	~/.venv/bin/python analyze.py results/$(RUN_ID)
