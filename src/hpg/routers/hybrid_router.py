"""System 1 first; System 2 only when System 1 is unsure (plan §5.2.3)."""

from __future__ import annotations

from hpg.routers.base import NONE_KEY, Router, RoutingDecision
from hpg.schema import AgentState, Edge


class HybridRouter(Router):
    name = "hybrid"

    def __init__(self, laya_router: Router, llm_router: Router, tau: float, gate: str = "confidence"):
        self.laya = laya_router
        self.llm = llm_router
        self.tau = tau
        self.gate = gate  # "confidence" (1 - normalised entropy) or "answer_confidence" (p of answer)

    def route(self, state: AgentState, node_id: str, edges: list[Edge], question: str) -> RoutingDecision:
        d = self.laya.route(state, node_id, edges, question)
        score = d.confidence if self.gate == "confidence" else (d.answer_confidence or 0.0)
        if score >= self.tau and d.raw_choice != NONE_KEY:
            return d.model_copy(update={"detail": {**d.detail, "gate": self.gate, "tau": self.tau}})
        f = self.llm.route(state, node_id, edges, question)
        return f.model_copy(
            update={
                "router": "llm_fallback",
                "latency_ms": d.latency_ms + f.latency_ms,
                "laya_confidence": score,
                "laya_target": d.target,
                "detail": {
                    **f.detail,
                    "laya_probs": d.probs,
                    "laya_raw_choice": d.raw_choice,
                    "gate": self.gate,
                    "tau": self.tau,
                    "fallback_reason": "laya_chose_none" if d.raw_choice == NONE_KEY else "below_tau",
                },
            }
        )
