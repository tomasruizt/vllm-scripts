SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
REMOTE ?= RH-nm-ore-b300-01
RUN_ID ?= glm53-flash-capture-pilot-20261019

.PHONY: fetch-results summarize
fetch-results:
	mkdir -p results
	ssh -T -o BatchMode=yes $(REMOTE) 'tar czf - --exclude="inputs.json" --exclude="profile_export_raw.jsonl" --exclude="gpu_telemetry_export.jsonl" -C ~/runs "$(RUN_ID)"' | tar xzf - -C results

summarize:
	~/.venv/bin/python summarize.py results/$(RUN_ID)
	duckdb -c "COPY (SELECT * FROM read_json_auto('results/$(RUN_ID)/metrics.json', format='array', maximum_object_size=1000000000)) TO 'results/$(RUN_ID)/metrics.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);"
	duckdb -f graph-coverage.sql
