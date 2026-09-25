"""Core data model: Procedural Graph (nodes/edges) and per-ticket agent state."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

END = "END"


class Node(BaseModel):
    id: str
    description: str
    executor: Literal["tool", "llm", "laya_check", "terminal", "route"]
    tool: str | None = None  # for executor == "tool"
    llm_task: str | None = None  # prompt template name for executor == "llm"
    laya_questions: dict | None = None  # for executor == "laya_check" (guardrails etc.)
    # Extension: the routing question asked at this node when it has several outgoing edges.
    question: str | None = None


class Edge(BaseModel):
    source: str
    relation: Literal["LEADS_TO"] = "LEADS_TO"
    target: str
    condition: str  # natural language: when to take this edge
    guidance: str  # natural language: how to execute the next step
    pitfalls: str  # natural language: mistakes to avoid
    # Extension (not in plan schema): bounded back-edges, e.g. final_review -> draft_reply retry.
    max_traversals: int | None = None
    on_exhausted: str | None = None  # node to go to instead once max_traversals is used up


class ProceduralGraph(BaseModel):
    name: str = "pg"
    nodes: list[Node]
    edges: list[Edge]
    start: str

    def node(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def out_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.source == node_id]

    def edge(self, source: str, target: str) -> Edge:
        for e in self.edges:
            if e.source == source and e.target == target:
                return e
        raise KeyError(f"{source}->{target}")


class ToolCall(BaseModel):
    tool: str
    args: dict[str, Any]
    result: Any


class AgentState(BaseModel):
    ticket_id: str
    ticket_text: str
    current_node: str
    path: list[str] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)  # short key facts from tool results (for Laya)
    tool_results: dict[str, Any] = Field(default_factory=dict)  # full tool outputs keyed by node id
    ledger: list[dict[str, Any]] = Field(default_factory=list)  # actions taken (refunds, escalations)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    draft_reply: str | None = None
    info_request: str | None = None
    investigation: str | None = None
    incoming_edge: Edge | None = None  # edge that led into current node (for guidance injection)
    decision_edges: list[Edge] = Field(default_factory=list)  # upstream multi-way decisions taken
    edge_traversals: dict[str, int] = Field(default_factory=dict)
    review_feedback: str | None = None
    routing_extra: str | None = None  # text shown to routers besides ticket+facts (e.g. draft under review)
