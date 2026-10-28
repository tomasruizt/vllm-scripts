SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
REMOTE ?= RH-h100-3
RUN_ID ?= hopper-speedbench-capture-20261020
REMOTE_ROOT ?= /data/tomasruizt/runs

.PHONY: fetch-results summarize watch
watch:
	deadline=$$((SECONDS + 10800)); \
	until ssh -T -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 $(REMOTE) 'cd "$(REMOTE_ROOT)/$(RUN_ID)"; timeout 10800 bash -c "until test -f launch-exit-code.txt; do sleep 5; done"'; do \
		test $$SECONDS -lt $$deadline || exit 1; \
		echo 'Connection lost, retrying result monitor in 30 seconds'; sleep 30; \
	done
	$(MAKE) fetch-results summarize

fetch-results:
	mkdir -p results
	ssh -T -o BatchMode=yes $(REMOTE) 'tar czf - --exclude="inputs.json" --exclude="profile_export_raw.jsonl" --exclude="gpu_telemetry_export.jsonl" -C "$(REMOTE_ROOT)" "$(RUN_ID)"' | tar xzf - -C results

summarize:
	~/.venv/bin/python collect.py results/$(RUN_ID)
	duckdb -c "COPY (SELECT * FROM read_json_auto('results/$(RUN_ID)/metrics.json', format='array')) TO 'results/$(RUN_ID)/metrics.parquet' (FORMAT PARQUET); COPY (SELECT * FROM read_json_auto('results/$(RUN_ID)/startup-metrics.json', format='array')) TO 'results/$(RUN_ID)/startup-metrics.parquet' (FORMAT PARQUET);"
