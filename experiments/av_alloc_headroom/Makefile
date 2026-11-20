SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
REMOTE ?= RH-nm-ohio-b200-01
RUN_ID ?= av-alloc-headroom-20261120
REMOTE_ROOT ?= runs
DATASETS ?= HumanEval math_reasoning writing
SSH := ssh -T -o BatchMode=yes -o ConnectTimeout=10
RUN := $(REMOTE_ROOT)/$(RUN_ID)

.PHONY: launch watch fetch-results analyze
# verify_all on GPU 2, policy on GPU 3 (GPUs 0, 1, 5 had CUDA context failures on 2026-11-19).
launch:
	$(SSH) $(REMOTE) 'mkdir -p "$(REMOTE_ROOT)" && mkdir "$(RUN)"'  # fails if the run exists: pick a new RUN_ID
	scp -q launch.sh make_prompts.py $(REMOTE):"$(RUN)/"
	$(SSH) $(REMOTE) 'cd "$(RUN)" && HF_HOME=/data/tomasruizt/huggingface ~/.venv/bin/python make_prompts.py prompts $(DATASETS)'
	$(SSH) $(REMOTE) 'cd "$(RUN)" && for m in verify_all:2 policy:3; do ROOT=$$PWD MODE=$${m%:*} GPU_ID=$${m#*:} DATASETS="$(DATASETS)" nohup bash launch.sh > launch-$${m%:*}.log 2>&1 < /dev/null & done; echo launched'

watch:
	$(SSH) $(REMOTE) 'cd "$(RUN)"; timeout 10800 bash -c "until test -f verify_all/exit-code.txt -a -f policy/exit-code.txt; do sleep 10; done"'

fetch-results:
	mkdir -p results
	$(SSH) $(REMOTE) 'tar czf - -C "$(REMOTE_ROOT)" --exclude=prompts "$(RUN_ID)"' | tar xzf - -C results

analyze:
	~/.venv/bin/python replay.py results/$(RUN_ID)
