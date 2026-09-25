import json

from hpg.schema import AgentState
from hpg.state_render import render_facts, render_state_for_laya


def wc(state: dict) -> int:  # stand-in tokenizer: whitespace tokens of the serialised dict
    return len(json.dumps(state).split())


def test_render_facts_order_and_bools():
    s = render_facts({"duplicate_charge": False, "order_found": True, "order_status": "delivered"})
    assert s == "order_found: true; order_status: delivered; duplicate_charge: false"


def test_short_ticket_untouched():
    st = AgentState(
        ticket_id="T", ticket_text="where is ORD-10001", current_node="x", facts={"order_found": True}
    )
    out = render_state_for_laya(st, 100, wc)
    assert out == {"ticket": "where is ORD-10001", "facts": "order_found: true"}


def test_long_ticket_truncated_facts_kept():
    text = " ".join(f"w{i}" for i in range(2000)) + " FINAL-ASK"
    facts = {"order_found": True, "days_since_delivery": 12, "duplicate_charge": False}
    st = AgentState(ticket_id="T", ticket_text=text, current_node="x", facts=facts)
    out = render_state_for_laya(st, 300, wc)
    assert out["facts"] == render_facts(facts)
    assert "[...]" in out["ticket"] and out["ticket"].startswith("w0") and out["ticket"].endswith("FINAL-ASK")
    assert wc(out) <= 300
