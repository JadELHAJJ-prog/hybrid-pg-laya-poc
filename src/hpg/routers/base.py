"""Router interface shared by the deterministic, Laya, LLM and hybrid routers (plan §5.2)."""

from __future__ import annotations

import string
from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

from hpg.schema import AgentState, Edge

RouterName = Literal["deterministic", "laya", "llm", "llm_fallback"]
NONE_KEY = "Z"
NONE_TEXT = "None of the listed conditions clearly applies."


class RoutingDecision(BaseModel):
    target: str
    confidence: float
    probs: dict[str, float]
    router: RouterName
    latency_ms: float
    # Extras for traces/eval (not in plan schema): why, and cost.
    answer_confidence: float | None = None  # P(chosen option) where available
    raw_choice: str | None = None  # option key the model picked (e.g. "Z")
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    laya_confidence: float | None = None  # for llm_fallback: the Laya confidence that triggered it
    laya_target: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


def option_keys(n: int) -> list[str]:
    keys = list(string.ascii_uppercase[:n])
    assert NONE_KEY not in keys, "too many edges for opaque letter keys"
    return keys


class Router(ABC):
    name: str

    @abstractmethod
    def route(self, state: AgentState, node_id: str, edges: list[Edge], question: str) -> RoutingDecision:
        """Pick one of `edges` (all leaving `node_id`). `question` is the node-level routing prompt."""
