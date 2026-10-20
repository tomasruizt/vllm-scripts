* Be concise in your answers
* Keep markdown reports / summaries VERY concise and to the point.
* Configure Codex in `~/.codex/config.toml` with `approval_policy = "on-request"` and `approvals_reviewer = "auto_review"` to route eligible command approvals to an automatic reviewer.
* Dont use python to make simple file edits. Please use your built in `apply_patch` command, or equivalent, instead.
* Latex doesn't render well in the terminal. Use an alternative.
* Use the python env in ~/.venv/bin/python. Use `uv` to install python 3.12.
* When creating PRs that are meant only for reviewing changes between two commits, don't mention the PRs or issues, because they become polluted with these references.
* structure code top-down. i.e. top in the file is high-level code/policy, while below is lower-level code that is used by the higher-level code.
* to use GPUs, reserve them over canhazgpu (https://github.com/russellb/canhazgpu).
* when writing md files, dont break lines unless there is a good reason, since e.g. on Github your lines will be wrapped. Let the IDE / website take care of rendering, so you don't have to. I suggest only breaking lines after a dot, or similar.
* in terms of writing stlye, don't use em-dashes, or ";", because I dont use them either. Use commas, dots, parenthesis and colons instead.
* Never rewrite the style of my prose, unless I explictily ask you to.
* You dont need to explain metrics like ITL, TPOT, etc. When writing summaries, assume a technical audience that is familiar with the domain.
* When available, use hf models from `RedHatAI`, e.g. speculators.
* Before downloading a hf model, check if the system has a mount where all users can share the model weights, and set it as a download location. Sometimes that path is not writeable, so check first.
* If you are benchmarking a model with multiple runs, try to spin up the vllm server only once, since starting the server is what takes often most time. However, in these cases we most usually want to disable prefix-caching, because otherwise the second and later runs would be artificially faster.
* For perf benchmarks prefer to use standard tooling. This includes e.g. `vllm bench sweep serve`, or ai-perf or similar things. These capture important metrics like TPOT, ITL, etc. If you are writing custom code and forget to capture metrics, we are wasting our fucking time.
* When summarizing the results of a benchmark, rather than stating in prose the vllm serve parameter, just provide the vllm serve command including all details. Same for the benchmarking command. Also when summarizing number like total request, output toks, etc, you can use bullet points rather than prose. This structure makes it clear that we are talking about a bunch of metrics/numbers, rather than at context.
* Often the benchmark summaries is meant to share as comments on github. There is no point in referencing runner files or absolute log paths that will not be available on Github. Please inline all relevant information or abstract it away.
* Sometimes we are trying to narrow down a minimal failure case for a model. It can make sense to keep the server running, and only launch small sets of requests to quickly iterate on a minimal failing workload.
* When analyzing the results of multiple bench runs, its best to gather all runs in a single long-format parquet file, so we can quickly query the results with duckdb cli / SQL.
* Pass `--disable-uvicorn-access-log` in general to reduce the log verbosity.
* Use the github cli `gh` when possible
* When making code changes, try to reduce the git diff, so reviews are easier for humans.
* In general, its better to use `chg run` rather than manual GPU reservations, unless strictly necessary.
* When doing sweeps over concurrencies and comparing methods, we need to be mindful of the runtime. In particular, running a workload at c8 is going to be much slower than at c256. We need to size the workload so as to spend ~30s per measurement (independent of batch size). Because we are running this accross K different c levels, and N different n_runs. Perhaps we can randomly subsample the workload for smaller c, and not compare absolute runtimes. We need to be smart about this!!! Think about the design before spending many hours running stuff!
* When estimating the runtime of sweeps, differentiate between the total sequential runtime across all variants, and the minimal runtime, if we could run all variants in parallel. Then connect this with our actually avilable resources for benchmarking.
* launch workloads on remote machines in asynchronous mode, so your main loop is never blocked, and you can continue to receive instructions and continue to perform other tasks.
* Keep your diffs small, the code vertically compact, and indentation small. Use functions to get rid of indentation. Use variables to avoid large multi-line calls that. Try to put the bodies of loops in a function to make it ovious what comes out of the loop. Use pure functions when possible. Staircase-shaped code that is with context managers, loops, list and object creations, etc. should be generally avoided. Use type hints whenever you can.
* in unit tests dont set monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1"), this should be set by the caller
* Don't poll remote jobs with fixed `sleep` calls. Wait on a condition in a single call instead, e.g. `timeout 600 bash -c "until grep -q DONE run.log; do sleep 5; done"`.
* On remote machines, only inspect logs with a short `tail -n 20` or `grep`. Never `cat` remote files. To analyze results, copy them locally with the experiment's `make fetch-results` target (create it if missing), and query them locally (e.g. with duckdb).
* When publishing a report (e.g. to my website), only publish the page and the files it links for download. Never publish raw data or log archives, and check every published file for local paths, usernames, hostnames and secrets first.
* Write generated outputs (reports, plots, CSVs, previews) only to the experiment's git-ignored folders: `results/`, `plots/` and `report/` (one subfolder per variant, e.g. `report/cg/`). Commit only code and docs.

beads (shared by agents of all providers):
* Track tasks and durable knowledge in beads (`bd`), so agents of different providers can collaborate. The database lives in the private repo `~/code/priv-knowledge-base/.beads`. `~/code/rh/.beads/redirect` and `vllm-scripts/.beads/redirect` (git-ignored) point there, so `bd` works from both.
* Never store beads data (issues, memories) in a public repo like vllm-scripts, and never configure a public repo as a beads sync remote.
* At session start run `bd prime` to load workflow context and memories (Claude Code runs it automatically via a SessionStart hook).
* Work items: `bd ready`, `bd show <id>`, `bd create "Title" --type task --priority 2`, `bd update <id> --claim`, `bd close <id>`.
* Memories: store with `bd remember "..."`, list with `bd memories`. Don't use provider-specific memory files (e.g. Claude's MEMORY.md).
* Instructions belong in AGENTS.md. Don't create CLAUDE.md or other provider-specific instruction files.

custom commands:
* vllm-install: install from precompiled wheels
* set-cuda-visible: set CUDA_VISIBLE_DEVICES env var depending on the GPUs reserved using canhazgpu.
* git aliases
l