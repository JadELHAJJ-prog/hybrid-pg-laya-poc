#!/usr/bin/env python3
"""PreToolUse guard: block writes that add paid/cloud LLM API usage (POC rule 12.1)."""
import json
import re
import sys

PATTERNS = [
    r"^\s*(import|from)\s+(openai|anthropic|cohere|mistralai|google\.generativeai|google\.genai|groq|together)\b",
    r"api\.openai\.com",
    r"api\.anthropic\.com",
    r"generativelanguage\.googleapis\.com",
    r"\b(OPENAI|ANTHROPIC|GEMINI|GOOGLE|COHERE|GROQ)_API_KEY\b",
]

data = json.load(sys.stdin)
ti = data.get("tool_input", {})
path = ti.get("file_path", "")
if not path.endswith((".py", ".toml", ".yaml", ".yml", ".ipynb", ".sh")) or "/.claude/" in path:
    sys.exit(0)
text = "\n".join(str(ti.get(k, "")) for k in ("content", "new_string"))
text += "\n".join(str(e.get("new_string", "")) for e in ti.get("edits", []) or [])
hits = [p for p in PATTERNS if re.search(p, text, re.MULTILINE)]
if hits:
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": f"POC rule: no paid/cloud APIs. Matched {hits} in {path}. Use local Ollama/Laya only.",
    }}))
sys.exit(0)
