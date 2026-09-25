#!/usr/bin/env bash
# SessionStart: inject current POC phase status + GPU state so every session starts oriented.
root="${CLAUDE_PROJECT_DIR:-$(pwd)}"
{
  echo "## POC status (from RESULTS.md phase headers)"
  grep -E '^## Phase' "$root/RESULTS.md" 2>/dev/null || echo "(RESULTS.md has no phase entries yet)"
  echo; echo "## Git"; git -C "$root" log --oneline -5 2>/dev/null
  echo; echo "## GPU"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null
  echo; echo "## Ollama loaded"; ollama ps 2>/dev/null | tail -n +2
} | jq -Rs '{hookSpecificOutput:{hookEventName:"SessionStart",additionalContext:.}}'
