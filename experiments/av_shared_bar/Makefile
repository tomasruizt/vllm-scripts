SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
# Data release of experiments/av_alloc_headroom (traces/<run>/{policy,verify_all}.parquet + cost tables).
TRACES ?= ../av_alloc_headroom/results/data-release/traces

.PHONY: analyze
analyze:
	~/.venv/bin/python shared_bar.py $(TRACES)
	~/.venv/bin/python report.py > /dev/null
	@echo "wrote results/report.md"
