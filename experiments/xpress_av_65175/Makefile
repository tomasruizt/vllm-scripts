SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := all

REMOTE ?= RH-h100-3
RUN_ID ?= xpress-av-65175d
REMOTE_ROOT ?= runs
SSH := ssh -T -o BatchMode=yes -o ConnectTimeout=10

# Defaults for `make watch TASK=... SUFFIX=...` and friends.
TASK ?= unit
SUFFIX ?=
DONE_FILE ?= launch-exit-code.txt
REPO_NAME ?= vllm-xpress-av
EXPERIMENT ?= pr
PORT ?= 8640
LOCAL_PYTHON ?= $(HOME)/.venv/bin/python

.PHONY: all unit e2e test perf-base perf-pr launch watch fetch-results report parquet

all: unit

# Each task gets its own remote run dir. mkdir fails if it exists: pick a new
# RUN_ID, never delete old runs.
unit: TASK := unit
unit: CHG_GPUS ?= --gpu-ids 0
unit: launch

e2e: TASK := e2e
e2e: CHG_GPUS ?= --gpu-ids 0
e2e: launch

test: TASK := test
test: CHG_GPUS ?= --gpu-ids 0
test: launch

perf-base: TASK := steady
perf-base: SUFFIX := -base
perf-base: REPO_NAME := vllm-xpress-av-base
perf-base: EXPERIMENT := base
perf-base: CHG_GPUS ?= --gpu-ids 0
perf-base: PORT := 8640
perf-base: MODES := off on
perf-base: launch

# Base and PR run at the same time, one GPU each: the AV on/off delta stays within a
# GPU, and the two run opposite mode orders so an order effect cancels across commits.
perf-pr: TASK := steady
perf-pr: SUFFIX := -pr
perf-pr: REPO_NAME := vllm-xpress-av
perf-pr: EXPERIMENT := pr
perf-pr: CHG_GPUS ?= --gpu-ids 6
perf-pr: PORT := 8641
perf-pr: MODES := on off
perf-pr: launch

launch:
	$(SSH) $(REMOTE) 'cd "$(REMOTE_ROOT)" && mkdir "$(RUN_ID)-$(TASK)$(SUFFIX)"'
	scp -q launch-task.sh av_e2e.py run-*.sh perf/run-steady.sh perf/steady-speedbench.yaml build_report.py test-overrides.patch $(REMOTE):"$(REMOTE_ROOT)/$(RUN_ID)-$(TASK)$(SUFFIX)/"
	$(SSH) $(REMOTE) 'cd "$(REMOTE_ROOT)/$(RUN_ID)-$(TASK)$(SUFFIX)" && ROOT=$$PWD TASK=$(TASK) REPO_NAME=$(REPO_NAME) EXPERIMENT=$(EXPERIMENT) PORT=$(PORT) MODES="$(MODES)" CHG_GPUS="$(CHG_GPUS)" nohup bash launch-task.sh > launch.log 2>&1 < /dev/null & echo launched $(TASK)$(SUFFIX) in $(REMOTE_ROOT)/$(RUN_ID)-$(TASK)$(SUFFIX)'

watch:
	$(SSH) $(REMOTE) 'cd "$(REMOTE_ROOT)/$(RUN_ID)-$(TASK)$(SUFFIX)"; timeout 14400 bash -c "until test -f $(DONE_FILE); do sleep 5; done"'
	$(MAKE) fetch-results TASK=$(TASK) SUFFIX="$(SUFFIX)"

fetch-results:
	mkdir -p results
	$(SSH) $(REMOTE) 'tar czf - -C "$(REMOTE_ROOT)" "$(RUN_ID)-$(TASK)$(SUFFIX)"' | tar xzf - -C results

# Steady-state AIPerf report: one tab per commit, medians over valid repeats,
# faint dots for individual runs, validity gates from build_report.py.
report:
	$(LOCAL_PYTHON) build_report.py --output report/perf \
		--run base:main:results/$(RUN_ID)-steady-base \
		--run pr:PR:results/$(RUN_ID)-steady-pr

parquet:
	$(LOCAL_PYTHON) build_parquet.py
	$(LOCAL_PYTHON) build_markdown.py
