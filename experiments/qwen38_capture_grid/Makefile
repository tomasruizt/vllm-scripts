SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
REMOTE ?= RH-nm-ore-b300-01
RUN_ID ?= qwen38-capture-grid-20261019
REMOTE_ROOT ?= /home/tomasruizt/runs/$(RUN_ID)
CONCURRENCIES ?= 32 128

.PHONY: fetch-results summarize combine plot recipe-plot summarize-startups
fetch-results:
	mkdir -p results
	ssh -T -o BatchMode=yes $(REMOTE) 'tar czf - --exclude="inputs.json" --exclude="profile_export_raw.jsonl" --exclude="gpu_telemetry_export.jsonl" -C "$$(dirname "$(REMOTE_ROOT)")" "$(RUN_ID)"' | tar xzf - -C results

summarize:
	~/.venv/bin/python summarize.py results/$(RUN_ID) --concurrencies $(CONCURRENCIES)
	duckdb -c "COPY (SELECT * FROM read_json_auto('results/$(RUN_ID)/metrics.json', format='array', maximum_object_size=1000000000)) TO 'results/$(RUN_ID)/metrics.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);"

combine:
	duckdb -c "COPY (SELECT * FROM read_parquet('results/qwen38-capture-grid*/metrics.parquet', union_by_name=true)) TO 'results/metrics-new.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);"
	mv results/metrics-new.parquet results/metrics.parquet

plot: combine
	duckdb -c "COPY (SELECT variant, concurrency, throughput, 1000.0 / mean_tpot_ms AS interactivity FROM read_parquet('results/metrics.parquet') WHERE valid ORDER BY capture_cap, concurrency) TO 'results/qwen38-capture-grid-20261019/plot.csv' (HEADER, DELIMITER ',');"
	~/.venv/bin/python plot.py

recipe-plot:
	duckdb -c "COPY (SELECT variant, repeat, concurrency, throughput, 1000.0 / mean_tpot_ms AS interactivity FROM read_parquet('results/$(RUN_ID)/metrics.parquet') WHERE valid ORDER BY capture_cap, concurrency, repeat) TO 'results/$(RUN_ID)/plot.csv' (HEADER, DELIMITER ',');"
	~/.venv/bin/python plot.py --root results/$(RUN_ID) --recipe

summarize-startups:
	~/.venv/bin/python collect-startups.py results/qwen38-recipe-grid-20261019 results/qwen38-recipe-restarts-20261020
	duckdb -c "COPY (SELECT * FROM read_json_auto('results/qwen38-recipe-restarts-20261020/metrics.json', format='array', maximum_object_size=1000000000)) TO 'results/qwen38-recipe-restarts-20261020/metrics.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);"
	duckdb -f startup-coverage.sql
