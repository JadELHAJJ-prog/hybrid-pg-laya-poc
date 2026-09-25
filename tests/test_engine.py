"""Engine + metrics integration using a gold-following router and a canned LLM (no models needed)."""

import json
from types import SimpleNamespace

from conftest import ROOT

from hpg.engine import Engine
from hpg.eval.metrics import final_action_from_trace, load_tickets, summarize
from hpg.routers.base import Router, RoutingDecision
from hpg.tracing import Tracer, read_trace


class FakeLLM:
    def chat(self, messages, **kw):
        return SimpleNamespace(content="Canned reply.", tool_calls=[], prompt_tokens=10, completion_tokens=5,
                               wall_ms=1.0, compute_ms=1.0, load_ms=0.0, logprobs=None, message=None)  # fmt: skip


class GoldRouter(Router):
    name = "gold"

    def __init__(self, gold, review_fails=0):
        self.gold = gold
        self.review_fails = review_fails

    def route(self, state, node_id, edges, question):
        if node_id == "final_review" and self.review_fails > 0:
            self.review_fails -= 1
            target = "draft_reply"
        else:
            target = self.gold[len(state.path)]
        return RoutingDecision(
            target=target, confidence=0.9, probs={target: 0.9}, router="laya", latency_ms=1.0
        )


def tickets():
    return [json.loads(x) for x in (ROOT / "data/tickets.jsonl").read_text().splitlines() if x]


def test_gold_router_reproduces_every_gold_path(tmp_path, graph, world):
    for t in tickets():
        eng = Engine(graph, world, GoldRouter(t["gold_path"]), FakeLLM())
        st = eng.run(t, Tracer(tmp_path / f"{t['ticket_id']}.jsonl"))
        assert st.path == t["gold_path"], t["ticket_id"]
        assert (
            final_action_from_trace(read_trace(tmp_path / f"{t['ticket_id']}.jsonl"))
            == t["gold_final_action"]
        )
    s = summarize(tmp_path, load_tickets(ROOT / "data/tickets.jsonl"))
    assert s["routing_acc"] == 1.0 and s["path_exact"] == 1.0 and s["final_action_acc"] == 1.0
    assert s["guard_recall_adv"] == 1.0 and s["guard_fpr_clean"] == 0.0 and s["errors"] == 0


def test_review_retry_then_escalate(tmp_path, graph, world):
    t = next(t for t in tickets() if t["gold_final_action"] == "refund_standard")
    eng = Engine(graph, world, GoldRouter(t["gold_path"], review_fails=2), FakeLLM())
    st = eng.run(t, Tracer(tmp_path / "x.jsonl"))
    assert (
        st.path[-5:]
        == ["draft_reply", "final_review", "draft_reply", "final_review", "escalate_human", "END"][-5:]
    )
    assert [e["action"] for e in st.ledger] == ["refund", "escalate"]
    notes = [r.get("note") for r in read_trace(tmp_path / "x.jsonl") if r["type"] == "route"]
    assert any(n and "exhausted" in n for n in notes)
