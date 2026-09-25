"""Validate data/tickets.jsonl against the graph and the mock world (plan §6, Phase 3)."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpg.gold import gold_path_for  # noqa: E402
from hpg.graph_loader import is_valid_path, load_graph  # noqa: E402
from hpg.tools import World  # noqa: E402

MIN_PER_EDGE = 8
# Edges whose traversal depends on generated text (reply review retry) cannot have a static gold label.
EXEMPT_EDGES = {("final_review", "draft_reply")}


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    g = load_graph(ROOT / cfg["paths"]["graph"])
    world = World.from_config(cfg, ROOT)
    tickets = [json.loads(line) for line in (ROOT / cfg["paths"]["tickets"]).read_text().splitlines() if line]
    errors: list[str] = []
    ids = [t["ticket_id"] for t in tickets]
    if len(ids) != len(set(ids)):
        errors.append("duplicate ticket ids")
    edge_count: Counter = Counter()
    for t in tickets:
        tid = t["ticket_id"]
        if not is_valid_path(g, t["gold_path"]):
            errors.append(f"{tid}: gold_path is not a valid graph walk: {t['gold_path']}")
        path, action = gold_path_for(world, t["gold_category"], t["text"])
        if path != t["gold_path"] or action != t["gold_final_action"]:
            errors.append(f"{tid}: gold inconsistent with orders.json (expected {path} / {action})")
        if t["split"] not in ("dev", "test"):
            errors.append(f"{tid}: bad split {t['split']!r}")
        for a, b in zip(t["gold_path"], t["gold_path"][1:], strict=False):
            edge_count[(a, b)] += 1
    for e in g.edges:
        key = (e.source, e.target)
        if key in EXEMPT_EDGES:
            continue
        if edge_count[key] < MIN_PER_EDGE:
            errors.append(f"edge {key} has only {edge_count[key]} gold examples (< {MIN_PER_EDGE})")
    splits = Counter(t["split"] for t in tickets)
    n = len(tickets)
    hard = sum("hard" in t.get("tags", []) for t in tickets)
    adv = sum("adversarial" in t.get("tags", []) for t in tickets)
    print(f"tickets={n} splits={dict(splits)} hard={hard} ({hard / n:.0%}) adversarial={adv} ({adv / n:.0%})")
    print(
        "edge coverage (min over non-exempt edges):",
        min(edge_count[(e.source, e.target)] for e in g.edges if (e.source, e.target) not in EXEMPT_EDGES),
    )
    if splits.get("dev") != 70 or splits.get("test") != 150:
        errors.append(f"split sizes {dict(splits)} != dev 70 / test 150")
    if errors:
        print(f"FAIL ({len(errors)} errors)")
        for e in errors[:50]:
            print("  -", e)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
