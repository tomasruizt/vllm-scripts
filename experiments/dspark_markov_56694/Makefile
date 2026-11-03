REMOTE ?=
RUN_ID ?= dspark-markov-56694-20261102-r2
REMOTE_ROOT ?= benchmarks/$(RUN_ID)
EXCLUDES = source client-venv data *.tar.gz output_fragments raw_records inputs.json profile_export_raw.jsonl

.PHONY: fetch-results
fetch-results:
	@test -n "$(REMOTE)" || { echo 'Set REMOTE to the benchmark host' >&2; exit 1; }
	mkdir -p results/$(RUN_ID)
	rsync -az $(foreach path,$(EXCLUDES),--exclude '$(path)') $(REMOTE):$(REMOTE_ROOT)/ results/$(RUN_ID)/
