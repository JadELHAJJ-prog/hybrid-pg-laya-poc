# Results log

Per-phase log for the Hybrid PG PoC. Numbers here are measured; see each phase for the command that produced them.

## Phase 0 – Scaffold (DONE 2026-09-25)
**Built:** uv project (`pyproject.toml`, Python 3.11.x, `uv.lock`), `config.yaml`, `hpg` Typer CLI skeleton, pytest.
Claude Code tooling: `CLAUDE.md`; `.claude/agents/` (trace-analyst, phase-auditor, laya-api-scout);
`.claude/skills/` (phase-closeout, run-experiment, edit-procedural-graph); hooks in `.claude/settings.json`
(PreToolUse paid-API guard, PostToolUse ruff format/fix, SessionStart status); graphify CLAUDE.md section +
post-commit/post-checkout hooks.
**Key numbers:** `uv run pytest -q` → smoke test passes; `uv run hpg --help` lists commands.
**Deviations from plan:** repo root is `jev-x-pg/` (not a `hybrid-pg-poc/` subfolder), per user choice.
pytest disables ROS pytest plugins that leak in via the host `PYTHONPATH` (`/opt/ros/jazzy`).
**Open issues:** none.

