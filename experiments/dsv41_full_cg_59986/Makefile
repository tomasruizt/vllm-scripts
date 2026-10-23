SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
REMOTE ?= RH-nm-ore-b300-01
RUN_ID ?= dsv41-full-cg-59986-20261023c
REMOTE_ROOT ?= /data/tomasruizt/runs
SSH := ssh -T -o BatchMode=yes -o ConnectTimeout=10

.PHONY: setup launch watch fetch-results summarize
setup:
	$(SSH) $(REMOTE) 'bash -s' < setup.sh

launch:
	$(SSH) $(REMOTE) 'mkdir "$(REMOTE_ROOT)/$(RUN_ID)"'  # fails if the run exists: pick a new RUN_ID, never delete old runs
	scp -q run.py launch.sh $(REMOTE):"$(REMOTE_ROOT)/$(RUN_ID)/"
	$(SSH) $(REMOTE) 'cd "$(REMOTE_ROOT)/$(RUN_ID)" && ROOT=$$PWD $(LAUNCH_ENV) nohup bash launch.sh > launch.log 2>&1 < /dev/null & echo launched'

watch:
	$(SSH) $(REMOTE) 'cd "$(REMOTE_ROOT)/$(RUN_ID)"; timeout 28800 bash -c "until test -f launch-exit-code.txt; do sleep 5; done"'
	$(MAKE) fetch-results summarize

fetch-results:
	mkdir -p results
	$(SSH) $(REMOTE) 'tar czf - --exclude="inputs.json" --exclude="profile_export_raw.jsonl" --exclude="gpu_telemetry_export.jsonl" -C "$(REMOTE_ROOT)" "$(RUN_ID)"' | tar xzf - -C results

summarize:
	python3 collect.py results/$(RUN_ID)
	duckdb -c "COPY (SELECT * FROM read_json_auto('results/$(RUN_ID)/metrics.json', format='array')) TO 'results/$(RUN_ID)/metrics.parquet' (FORMAT PARQUET)"
