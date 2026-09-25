---
name: edit-procedural-graph
description: Use when adding, removing, or rewording nodes/edges in graphs/*.yaml (the RefundDesk Procedural Graph), including edge condition/guidance/pitfalls text or self-evolution edits (Phase 8).
---

# Editing the Procedural Graph

The graph is data. Engine code must not special-case node ids beyond `start` and `END`.

Rules:
1. Every edge needs `condition`, `guidance`, and `pitfalls` text. Write conditions as mutually exclusive, state-observable facts, because Laya scores them as `choice` options against the rendered state.
2. Conditions may only refer to facts that `render_state_for_laya` actually emits. If a condition needs a new fact, add it to the renderer and test it.
3. After an edit, run:
   - `uv run python -c "from hpg.graph_loader import load_graph; load_graph('graphs/refunddesk_v1.yaml')"` (validation: endpoints exist, all nodes reach END, cycles only where declared)
   - `uv run python scripts/validate_dataset.py` (gold paths still valid, ≥8 examples per edge)
   - `uv run pytest -q`
4. Changing edge text changes routing, so re-tune tau on `dev` and never on `test`.
5. For Phase 8 self-evolution, keep an edit only if dev routing accuracy does not drop. Log rejected edits in `runs/evolution/rejected.jsonl`.
