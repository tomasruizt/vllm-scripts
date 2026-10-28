SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
REMOTE ?= RH-h100-1
RUN_ID ?= dsv4-small-av-prefill-20261023
REMOTE_ROOT ?= runs
SSH := ssh -T -o BatchMode=yes -o ConnectTimeout=10

.PHONY: launch watch fetch-results
launch:
	$(SSH) $(REMOTE) 'mkdir "$(REMOTE_ROOT)/$(RUN_ID)"'  # fails if the run exists: pick a new RUN_ID, never delete old runs
	scp -q run.py launch.sh $(REMOTE):"$(REMOTE_ROOT)/$(RUN_ID)/"
	$(SSH) $(REMOTE) 'cd "$(REMOTE_ROOT)/$(RUN_ID)" && ROOT=$$PWD nohup bash launch.sh > launch.log 2>&1 < /dev/null & echo launched'

watch:
	$(SSH) $(REMOTE) 'cd "$(REMOTE_ROOT)/$(RUN_ID)"; timeout 3600 bash -c "until test -f launch-exit-code.txt; do sleep 5; done"'
	$(MAKE) fetch-results

fetch-results:
	mkdir -p results
	$(SSH) $(REMOTE) 'tar czf - --exclude="inputs.json" --exclude="profile_export_raw.jsonl" --exclude="gpu_telemetry_export.jsonl" -C "$(REMOTE_ROOT)" "$(RUN_ID)"' | tar xzf - -C results
