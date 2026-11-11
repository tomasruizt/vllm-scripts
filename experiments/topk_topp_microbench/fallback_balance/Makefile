SHELL := /bin/bash
# Fetch the run dir's results into ../results/fallback-balance/$(GPU), then rebuild the parquet.
# make fetch-results GPU=mi355x HOST=g12-34 SSH_CONFIG=...   (NVIDIA hosts: no SSH_CONFIG)
GPU ?= mi355x
HOST ?= g12-34
SSH_CONFIG ?=
RUN_DIR ?= runs/fallback-balance
OUT := ../results/fallback-balance
SCP := scp -rq $(if $(SSH_CONFIG),-F $(SSH_CONFIG))

.PHONY: fetch-results parquet
fetch-results:
	mkdir -p $(OUT)/$(GPU)
	$(SCP) $(HOST):$(RUN_DIR)/results/. $(OUT)/$(GPU)/
	$(MAKE) parquet

parquet:
	cd $(OUT) && duckdb -c "COPY (SELECT * FROM read_json_auto('*/*/*_r*.jsonl', filename=true, union_by_name=true)) TO 'raw.parquet'" && duckdb -c "$$(cat ../../fallback_balance/long.sql)"
