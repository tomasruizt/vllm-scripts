SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
REMOTE ?= RH-nm-ore-b300-01
RUN_ID ?= glm53-flash-capture-pilot-20261019

.PHONY: fetch-results summarize plot watch-cost summarize-cost
watch-cost:
	ssh -T -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 $(REMOTE) 'cd ~/runs/$(RUN_ID); timeout 14400 bash -c "until test -f job-exit-code.txt; do sleep 5; done"'
	$(MAKE) fetch-results summarize-cost

summarize-cost:
	duckdb -c "COPY (SELECT *,regexp_extract(filename, 'rank([0-9]+)', 1)::INTEGER AS rank FROM read_json('results/$(RUN_ID)/probe*-rank*.jsonl',filename=true)) TO 'results/$(RUN_ID)/steps.parquet' (FORMAT PARQUET);"
	duckdb -c "SELECT rank,mode,dispatched_mode,padded_tokens,prefill_reqs>0 AS has_prefill,count(*) AS n,round(median(wall_ms),3) AS wall_ms,round(median(gpu_ms),3) AS gpu_ms FROM 'results/$(RUN_ID)/steps.parquet' WHERE NOT warmup GROUP BY ALL ORDER BY rank,padded_tokens,has_prefill,mode;" > results/$(RUN_ID)/cost-summary.txt

plot:
	duckdb -c "COPY (SELECT variant, repeat, concurrency, throughput, 1000.0 / mean_tpot_ms AS interactivity FROM 'results/$(RUN_ID)/metrics.parquet' WHERE valid ORDER BY capture_cap, concurrency, repeat) TO 'results/$(RUN_ID)/plot.csv' (HEADER);"
	~/.venv/bin/python plot.py --root results/$(RUN_ID)

fetch-results:
	mkdir -p results
	ssh -T -o BatchMode=yes $(REMOTE) 'tar czf - --exclude="inputs.json" --exclude="profile_export_raw.jsonl" --exclude="gpu_telemetry_export.jsonl" -C ~/runs "$(RUN_ID)"' | tar xzf - -C results

summarize:
	~/.venv/bin/python summarize.py results/$(RUN_ID)
	duckdb -c "COPY (SELECT * FROM read_json_auto('results/$(RUN_ID)/metrics.json', format='array', maximum_object_size=1000000000)) TO 'results/$(RUN_ID)/metrics.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);"
	duckdb -f graph-coverage.sql
