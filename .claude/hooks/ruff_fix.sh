#!/usr/bin/env bash
# PostToolUse: format + autofix edited Python files with the project's ruff; report leftovers to Claude.
f=$(jq -r '.tool_response.filePath // .tool_input.file_path // empty')
[[ "$f" == *.py && -f "$f" ]] || exit 0
root="${CLAUDE_PROJECT_DIR:-$(pwd)}"
ruff="$root/.venv/bin/ruff"; [[ -x "$ruff" ]] || ruff=$(command -v ruff) || exit 0
"$ruff" format -q "$f" >/dev/null 2>&1
"$ruff" check -q --fix "$f" >/dev/null 2>&1
out=$("$ruff" check -q "$f" 2>&1 | head -20)
if [[ -n "$out" ]]; then
  jq -n --arg o "$out" '{hookSpecificOutput:{hookEventName:"PostToolUse",additionalContext:("ruff issues remaining:\n"+$o)}}'
fi
exit 0
