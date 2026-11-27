# Profiling of GLM-5.3-Flash on vLLM: a step-time microbenchmark (forward.sh) and a trial of the NVIDIA Nsight Perf SDK
# for AI under an AgentX workload (nsight.sh). See README.md.
# Reuses the model disk, CPU pod and AgentX client of ../glm53_agentx: run `make box prepare` there first, with the same
# KUBE_CONTEXT, NS, NODE and PREFIX. For `make nsight`, NS must be a namespace whose service account PROFILER_SA may add
# CAP_PERFMON.
KUBE_CONTEXT ?=
NS ?=
NODE ?= b200-02
PREFIX ?= glm53
PROFILER_SA ?= perfmon
K := kubectl $(if $(KUBE_CONTEXT),--context $(KUBE_CONTEXT)) $(if $(NS),-n $(NS))

# nsight.sh: one AgentX point.
CONC ?= 8
DURATION ?= 300
WARMUP ?= 1
RUN_ID ?= nsight-c$(CONC)-1
# forward.sh: GPUs (= TP size) and the model. Smoke test on 1 GPU with the small clone, which has no MTP weights:
#   make forward GPUS=1 MEM=128Gi MODEL=inference-optimization/GLM-5.3-Flash-2.6B-A2.3B-small REVISION= LOAD_FORMAT=dummy
GPUS ?= 4
MEM ?= 512Gi
MODEL ?= zai-org/GLM-5.3-Flash
REVISION ?= 46a050bec13002ad29ee795c16e61cf4fa3c4a9c
LOAD_FORMAT ?= auto

SCRIPTS := forward.sh nsight.sh prepare.sh ../glm53_agentx/bench.sh
COMMON_VARS := s/__PREFIX__/$(PREFIX)/g; s/__NODE__/$(NODE)/; s/__RUN_ID__/$(RUN_ID)/g
NSIGHT_VARS := s/__PROFILER_SA__/$(PROFILER_SA)/; s/__CONC__/$(CONC)/; s/__DURATION__/$(DURATION)/; s/__WARMUP__/$(WARMUP)/
FORWARD_VARS := s/__GPUS__/$(GPUS)/g; s/__MEM__/$(MEM)/g; s|__MODEL__|$(MODEL)|; s/__REVISION__/$(REVISION)/; s/__LOAD_FORMAT__/$(LOAD_FORMAT)/

scripts:
	$(K) create configmap $(PREFIX)-profiling-scripts $(addprefix --from-file=,$(SCRIPTS)) --dry-run=client -o yaml | $(K) apply -f -

prepare:
	$(K) exec -i $(PREFIX)-box -- bash < prepare.sh

forward: scripts
	$(K) delete pod $(PREFIX)-forward-$(RUN_ID) --ignore-not-found --wait=true
	sed '$(COMMON_VARS); $(FORWARD_VARS)' forward.yaml | $(K) apply -f -

nsight: scripts
	$(K) delete pod $(PREFIX)-nsight-$(RUN_ID) --ignore-not-found --wait=true
	sed '$(COMMON_VARS); $(NSIGHT_VARS)' nsight.yaml | $(K) apply -f -

status:
	$(K) get pods --no-headers | grep -- '-$(RUN_ID) '; $(K) exec $(PREFIX)-box -- bash -c 'cd /models/results/$(RUN_ID) && cat startup.txt status.txt 2> /dev/null; tail -n 3 *.log'

# Logs, forward.json and the parquet export, without the DuckDB database and raw request records, which are large.
fetch-results:
	mkdir -p results
	$(K) exec $(PREFIX)-box -- tar -C /models/results --exclude='*raw*' --exclude='*.jsonl' --exclude='*.duckdb' -czf - $(RUN_ID) | tar -C results -xzf -
