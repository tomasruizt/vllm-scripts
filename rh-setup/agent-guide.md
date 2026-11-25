# Agent guide

Lessons from past sessions. Read the section that matches the task at hand.

## Spec-decode glossary

Terms that agents got wrong in past sessions. Re-verify entries older than a few weeks against the code, and update them here.

* K = `num_speculative_tokens` = draft depth. T = 1 + K = tokens the target scores per step.
* DFlash convention: K = block_size - 1. DSpark convention: K = block_size, so a block_size=8 drafter runs at T=9 (vLLM #65874). Many kernels cap T at 8. (checked 2026-11-23)
* Adaptive verification (AV) does not change K. It decides how many of the K drafted tokens the target verifies. "Drafted", "verified" and "accepted" are three different counts.
* Do not assume a proposer drafts greedily. Read the proposer code for the model at hand.
* `--use-replayssm` covers two mechanisms. ReplaySSM (#48018): checkpoint plus ring buffer of recent inputs, decode and spec-decode, on main only for Mamba2. RecoverSSM (#51855): spec-decode only, reconstructs the accepted state after sampling, on main only for KDA. (checked 2026-11-16)
* A ReplaySSM PR for a family does not imply support for every model of that family, nor for both Hopper and Blackwell.

## Before publishing anything

Applies to GitHub issues, PR descriptions, comments, Jira tickets, Slack messages and research notes.

* Draft to a markdown file and link it. Post only after an explicit go-ahead, then show the link.
* Fact-check before showing the draft: verify every number, PR reference and "X supports Y" claim against the source. Reading a PR title is not reading the PR. Say which claims were verified by running code, by reading code, or inferred.
* First sentence: what we wanted to find out. Second: what we found. Then evidence.
* Half the length of the first instinct. No inventory of what was done, no caveat lists.
* A PR description says what kind of change it is (bugfix, refactor, behavior change), with a before/after table if behavior changes.
* Never frame a negative result as a flaw in a colleague's idea. Credit joint work.

## Iterating on code with the user

* If the user describes a shape (a function name, a predicate, a call site), implement exactly that shape. If it cannot work, say why in one sentence and stop.
* If the shape is open, first show a sketch of signatures and the call site, without implementation.
* Change only what was asked. No new tests, fallbacks, predicates or config knobs unless asked.
* Do not run test suites, commit or push between edits unless asked.
* When the proper mechanism is missing, do not improvise a workaround. Look at how sibling projects solve it, and propose that.
* A missing package or CLI is not a finding. Install it and continue.

## Hand-offs

* After a push, PR, issue or ticket: show the link. After a plot or report: link the local file.
* When a background job finishes, report without being asked.
* At the end of a task, list uncommitted changes, worktrees and GPU reservations from this session, and propose what to commit, close and release.
