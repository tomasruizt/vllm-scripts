#!/usr/bin/env bash
# Measure friction in Claude Code sessions: how often the user has to correct, poll or repeat.
#   ./run.sh SINCE UNTIL [OUT_DIR]     e.g. ./run.sh 2026-11-16 2026-11-25T13:30
# Writes OUT_DIR/ev.db (events), prompts.txt (all human prompts) and errors.txt (failed tool calls).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
SESSIONS=~/.claude/projects

main() {
  local since=$1 until=$2 out=${3:-$(mktemp -d)}
  mkdir -p "$out"
  extract_events "$since" "$until" > "$out/ev.jsonl"
  cd "$out"
  duckdb ev.db -cmd ".maxwidth 250" < "$HERE/report.sql"
  echo "outputs in $out"
}

extract_events() {
  local since=$1 until=$2 file
  find "$SESSIONS" -name '*.jsonl' -newermt "${since%%T*}" | while read -r file; do
    jq -c --arg since "$since" --arg until "$until" -f "$HERE/extract.jq" "$file" 2>/dev/null || true
  done
}

main "$@"
