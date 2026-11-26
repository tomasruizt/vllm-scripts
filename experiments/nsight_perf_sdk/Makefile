# Trial of the NVIDIA Nsight Perf SDK for AI on a vLLM server under an AgentX workload. See README.md.
# Reuses the model disk, CPU pod and AgentX client of ../glm53_agentx: run `make box prepare` there first, with the same
# KUBE_CONTEXT, NS, NODE and PREFIX. NS must be a namespace whose service account PROFILER_SA may add CAP_PERFMON.
KUBE_CONTEXT ?=
NS ?=
NODE ?= b200-02
PREFIX ?= glm53
PROFILER_SA ?= perfmon
K := kubectl $(if $(KUBE_CONTEXT),--context $(KUBE_CONTEXT)) $(if $(NS),-n $(NS))

CONC ?= 8
DURATION ?= 300
WARMUP ?= 1
RUN_ID ?= nsight-c$(CONC)-1

SCRIPTS := nsight.sh prepare.sh ../glm53_agentx/bench.sh
VARS := s/__PREFIX__/$(PREFIX)/g; s/__NODE__/$(NODE)/; s/__PROFILER_SA__/$(PROFILER_SA)/; s/__RUN_ID__/$(RUN_ID)/g; s/__CONC__/$(CONC)/; s/__DURATION__/$(DURATION)/; s/__WARMUP__/$(WARMUP)/

scripts:
	$(K) create configmap $(PREFIX)-nsight-scripts $(addprefix --from-file=,$(SCRIPTS)) --dry-run=client -o yaml | $(K) apply -f -

prepare:
	$(K) exec -i $(PREFIX)-box -- bash < prepare.sh

run: scripts
	$(K) delete pod $(PREFIX)-nsight-$(RUN_ID) --ignore-not-found --wait=true
	sed '$(VARS)' nsight.yaml | $(K) apply -f -

status:
	$(K) get pod $(PREFIX)-nsight-$(RUN_ID) --no-headers; $(K) exec $(PREFIX)-box -- bash -c 'cd /models/results/$(RUN_ID) && cat startup.txt status.txt 2> /dev/null; tail -n 3 export.log; tail -n 5 client.log'

# The parquet export and logs, without the DuckDB database and raw request records, which are large.
fetch-results:
	mkdir -p results
	$(K) exec $(PREFIX)-box -- tar -C /models/results --exclude='*raw*' --exclude='*.jsonl' --exclude='*.duckdb' -czf - $(RUN_ID) | tar -C results -xzf -
