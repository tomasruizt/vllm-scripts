# Session friction

Measures how often I have to correct, poll or repeat myself in Claude Code sessions, so that changes to the agent setup can be judged by evidence.

```
./run.sh 2026-11-16 2026-11-25T13:30
```

It prints keyword-based friction rates per 100 prompts and tool error counts, and writes `prompts.txt` (all human prompts) and `errors.txt` (failed tool calls) for reading. The keyword rates catch only part of the real friction, so compare windows and read the prompts.

## Baseline

Window 2026-11-16 to 2026-11-25, before the `bd prime` truncation fix (`../bd-memory-digest`, vllm-scripts ccf08f7). Until then agents saw only a 2KB preview of the beads memories at session start.

* sessions: 30
* prompts: 718
* failed tool calls: 105 of 3.5k, mostly harmless non-zero exit codes

| category | n | per 100 prompts |
|---|---|---|
| commit_push | 24 | 3.3 |
| show_me_link | 20 | 2.8 |
| too_verbose | 19 | 2.6 |
| status_poll | 18 | 2.5 |
| overbuilt | 17 | 2.4 |
| verify | 11 | 1.5 |
| frustration | 10 | 1.4 |
| worktree | 9 | 1.3 |

Rerun on a later window of similar length and compare the rates. `commit_push` and `status_poll` include normal requests, so the signal is in `too_verbose`, `overbuilt`, `verify` and `frustration`.
