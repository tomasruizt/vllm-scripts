SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
# The data release of speculators#1430 (traces/<run>/<mode>.parquet and cost tables).
DATA ?= ../av_alloc_headroom/results/data-release

.PHONY: analyze
analyze:
	~/.venv/bin/python stale.py $(DATA)
	duckdb < summary.sql | tee results/summary.txt
