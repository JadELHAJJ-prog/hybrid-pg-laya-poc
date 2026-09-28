# Hybrid System-1 / System-2 agent on a Procedural Graph (PoC)

The original PoC plan is kept locally as `POC PLAN_5282.md` (not published; gitignored). Read it if present. Progress is logged per phase in `RESULTS.md`.

## Non-negotiables (plan §12)
- $0 and local only: Ollama LLM + Laya. No paid or cloud APIs, API keys, or hosted inference. A PreToolUse hook blocks paid-API imports.
- Verify, don't guess: read installed package source (use the `laya-api-scout` agent) and check real model tags. When the library disagrees with the plan, follow the library and log the deviation in RESULTS.md.
- Never tune on `test`. Tau, temperatures, checkpoint choice, and prompt/edge wording use `dev` only.
- Honest reporting: RESULTS.md holds only measured numbers, including negative results.
- One commit per phase, with tests passing (use the `phase-closeout` skill).
- Keep the handoff visible: every trace step records which layer decided it and why.

## Commands
- Env: `uv sync` (Python 3.11, `.venv/`). Tests: `uv run pytest -q`. Lint: `uv run ruff check .`
- CLI: `uv run hpg --help`, `uv run hpg run --ticket T0001 --router hybrid`, `uv run hpg eval --all`
- Dataset check: `uv run python scripts/validate_dataset.py`. VRAM: `uv run python scripts/vram_probe.py`

## Layout
`src/hpg/` engine (schema, graph_loader, state_render, tools, routers/, llm_nodes, checks, engine, tracing, eval/, cli) ·
`graphs/` PG YAML (data, not code) · `data/` mock world + tickets · `runs/` traces (gitignored) · `scripts/` probes/validators.

## Hardware
RTX 4060 Laptop, 8 GB VRAM, shared by Laya and Ollama. Cap `num_ctx` at 8192. If the LLM spills to CPU, apply the fallbacks in plan §2 in order and record which one was used.

## Project Claude tooling
- Agents: `trace-analyst` (failure analysis), `phase-auditor` (acceptance gate), `laya-api-scout` (verified Laya API).
- Skills: `phase-closeout`, `run-experiment`, `edit-procedural-graph`.
- Hooks (.claude/settings.json): paid-API guard, ruff auto-fix on .py edits, SessionStart status summary.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
