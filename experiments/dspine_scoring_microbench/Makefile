SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
REMOTE ?= RH-nm-frk-h100-01-preserve
RUN_ID ?= dspine-scoring-h100-20261113
REMOTE_ROOT ?= runs
SSH := ssh -T -o BatchMode=yes -o ConnectTimeout=10

.PHONY: launch watch fetch-results
launch:
	$(SSH) $(REMOTE) 'mkdir -p "$(REMOTE_ROOT)" && mkdir "$(REMOTE_ROOT)/$(RUN_ID)"'  # fails if the run exists: pick a new RUN_ID
	scp -q bench.py launch.sh $(REMOTE):"$(REMOTE_ROOT)/$(RUN_ID)/"
	$(SSH) $(REMOTE) 'cd "$(REMOTE_ROOT)/$(RUN_ID)" && ROOT=$$PWD VENV="$(VENV)" BENCH_ARGS="$(BENCH_ARGS)" setsid nohup bash launch.sh > launch.log 2>&1 < /dev/null & echo launched'

watch:
	$(SSH) $(REMOTE) 'cd "$(REMOTE_ROOT)/$(RUN_ID)"; timeout 1800 bash -c "until test -f launch-exit-code.txt; do sleep 5; done"'
	$(MAKE) fetch-results

fetch-results:
	mkdir -p results
	$(SSH) $(REMOTE) 'tar czf - -C "$(REMOTE_ROOT)" "$(RUN_ID)"' | tar xzf - -C results
