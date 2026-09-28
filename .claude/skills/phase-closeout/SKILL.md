---
name: phase-closeout
description: Use when finishing any phase (0-8) of the Hybrid PG PoC, before telling the user a phase is done or committing it. Runs the acceptance checks, appends the phase entry to RESULTS.md, and makes the per-phase commit.
---

# Phase close-out (Hybrid PG PoC)

A phase is not done until every step below has evidence.

1. **Acceptance criteria**: re-read the phase's "Done when" in the local PoC plan (`POC PLAN_5282.md`, if present) §10. Run each check and keep the output.
2. **Tests**: `uv run pytest -q` must pass. Run `uv run ruff check src tests scripts` as well.
3. **Audit**: dispatch the `phase-auditor` agent with the phase number. Fix any blocking issues it finds, then re-run it.
4. **RESULTS.md**: append a section with exactly this header shape (the SessionStart hook greps for it):
   ```
   ## Phase N – <name> (DONE YYYY-MM-DD)
   **Built:** ...
   **Key numbers:** ... (measured, with command that produced them)
   **Deviations from plan:** ...
   **Open issues:** ...
   ```
   Report negative results as measured. Never put a number here that a command did not produce.
5. **Commit**: `git add -A && git commit -m "phase N: <summary>"`, ending the message with the Co-Authored-By trailer. The graphify post-commit hook rebuilds `graphify-out/` automatically.
6. Tell the user in 3-5 lines: what was built, the key numbers, and any deviation.
