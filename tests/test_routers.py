"""Router logic with fake models (no GPU / Ollama needed)."""

import json
from types import SimpleNamespace

from hpg.routers.base import RoutingDecision
from hpg.routers.hybrid_router import HybridRouter
from hpg.routers.laya_router import LayaRouter
from hpg.routers.llm_router import LLMRouter, build_prompt
from hpg.schema import AgentState


class FakeTok:
    mask_token = "[MASK]"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


class FakeAgent:
    def __init__(self, probs):
        self.probs = probs
        self.cfg = {"max_len": 1024, "head_max_len": 256}
        self.tok = FakeTok()
        self.temperature_by_options = {}
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        q = next(iter(questions))
        crit = questions[q]["criteria"]
        p = {k: self.probs.get(k, 0.0) for k in crit}
        choice = max(p, key=p.get)
        return {"answers": {q: {"choice": choice, "probabilities": p, "confidence": 0.3,
                                "answer_confidence": p[choice]}}, "usage": {"input_tokens": 10}}  # fmt: skip


def st():
    return AgentState(ticket_id="T", ticket_text="charged twice for ORD-10031", current_node="classify",
                      facts={"order_id_in_ticket": "ORD-10031"})  # fmt: skip


def test_laya_router_opaque_keys_and_mapping(graph):
    agent = FakeAgent({"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.05, "Z": 0.05})
    r = LayaRouter(agent, graph)
    edges = graph.out_edges("classify")
    d = r.route(st(), "classify", edges, "What kind?")
    crit = agent.calls[0][1]["route"]["criteria"]
    assert list(crit) == ["A", "B", "C", "D", "Z"]
    assert crit["A"] == edges[0].condition
    assert d.target == edges[0].target and d.router == "laya" and d.answer_confidence == 0.7
    assert set(agent.calls[0][0]) == {"ticket", "facts"}


def test_laya_router_none_option_picks_best_real_edge(graph):
    agent = FakeAgent({"A": 0.1, "B": 0.3, "Z": 0.6})
    d = LayaRouter(agent, graph).route(st(), "lookup_order", graph.out_edges("lookup_order"), "q")
    assert d.raw_choice == "Z" and d.target == graph.out_edges("lookup_order")[1].target


def test_laya_router_truncates_ticket_not_facts(graph):
    agent = FakeAgent({"A": 1.0})
    s = st()
    s.ticket_text = " ".join(["blah"] * 5000)
    LayaRouter(agent, graph).route(s, "lookup_order", graph.out_edges("lookup_order"), "q")
    sent = agent.calls[0][0]
    assert sent["facts"] == "order_id_in_ticket: ORD-10031" and "[...]" in sent["ticket"]
    assert len(json.dumps(sent).split()) <= 1024 - 256


class FakeLLM:
    def __init__(self, choice):
        self.choice = choice
        self.last = None

    def chat(self, messages, **kw):
        self.last = (messages, kw)
        lp = SimpleNamespace(token=self.choice, logprob=-0.1,
                             top_logprobs=[SimpleNamespace(token=self.choice, logprob=-0.1),
                                           SimpleNamespace(token="B", logprob=-2.5)])  # fmt: skip
        return SimpleNamespace(content=json.dumps({"choice": self.choice}), prompt_tokens=100, completion_tokens=9,
                               compute_ms=50.0, load_ms=1000.0, logprobs=[lp])  # fmt: skip


def test_llm_router_parses_choice_and_confidence(graph):
    edges = graph.out_edges("classify")
    llm = FakeLLM("C")
    d = LLMRouter(llm).route(st(), "classify", edges, "What kind?")
    assert d.target == edges[2].target and d.router == "llm" and 0.85 < d.confidence < 1.0
    assert llm.last[1]["format"]["properties"]["choice"]["enum"] == ["A", "B", "C", "D", "Z"]
    assert llm.last[1]["think"] is False
    assert d.llm_prompt_tokens == 100


def test_llm_prompt_contains_conditions_and_facts(graph):
    edges = graph.out_edges("policy_check")
    p, keys = build_prompt(st(), "policy_check", edges, "which outcome?")
    assert keys == ["A", "B", "C"] and all(e.condition in p for e in edges) and "ORD-10031" in p


class Const:
    def __init__(self, target, conf, raw="A"):
        self.d = RoutingDecision(target=target, confidence=conf, answer_confidence=conf, probs={}, router="laya",
                                 latency_ms=5, raw_choice=raw)  # fmt: skip
        self.n = 0

    def route(self, *a):
        self.n += 1
        return self.d


def test_hybrid_gate():
    llm = Const("y", 0.99)
    h = HybridRouter(Const("x", 0.8), llm, tau=0.5)
    assert h.route(None, "n", [], "q").target == "x" and llm.n == 0
    h = HybridRouter(Const("x", 0.3), llm, tau=0.5)
    d = h.route(None, "n", [], "q")
    assert d.target == "y" and d.router == "llm_fallback" and d.laya_target == "x" and d.laya_confidence == 0.3
    h = HybridRouter(Const("x", 0.9, raw="Z"), llm, tau=0.5)
    assert h.route(None, "n", [], "q").router == "llm_fallback"
