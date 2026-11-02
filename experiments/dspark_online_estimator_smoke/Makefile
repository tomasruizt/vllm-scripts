SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
REMOTE ?= RH-h100-3
RUN_ID ?= dspark-online-estimator-smoke-20261030
REMOTE_ROOT ?= runs
CHG_GPUS ?= --gpus 1
VARIANTS ?= auto,trained,online,no_av,online_argmax,trained_argmax,xpress,xpress_av
SSH := ssh -T -o BatchMode=yes -o ConnectTimeout=10

.PHONY: launch watch fetch-results
launch:
	$(SSH) $(REMOTE) 'mkdir -p "$(REMOTE_ROOT)" && mkdir "$(REMOTE_ROOT)/$(RUN_ID)"'  # fails if the run exists: pick a new RUN_ID, never delete old runs
	scp -q smoke.py launch.sh $(REMOTE):"$(REMOTE_ROOT)/$(RUN_ID)/"
	$(SSH) $(REMOTE) 'cd "$(REMOTE_ROOT)/$(RUN_ID)" && ROOT=$$PWD CHG_GPUS="$(CHG_GPUS)" VARIANTS="$(VARIANTS)" nohup bash launch.sh > launch.log 2>&1 < /dev/null & echo launched'

watch:
	$(SSH) $(REMOTE) 'cd "$(REMOTE_ROOT)/$(RUN_ID)"; timeout 3600 bash -c "until test -f launch-exit-code.txt; do sleep 5; done"'
	$(MAKE) fetch-results

fetch-results:
	mkdir -p results
	$(SSH) $(REMOTE) 'tar czf - -C "$(REMOTE_ROOT)" "$(RUN_ID)"' | tar xzf - -C results
