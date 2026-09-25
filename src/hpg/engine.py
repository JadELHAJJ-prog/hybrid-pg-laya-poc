"""Graph walker: executes nodes, routes edges, enforces max-steps and bounded retries (plan §5)."""

from __future__ import annotations

import time
from typing import Any

from hpg.llm import LLM
from hpg.llm_nodes import run_ask_for_info, run_draft_reply, run_investigate_technical
from hpg.routers.base import Router, RoutingDecision
from hpg.schema import END, AgentState, Edge, Node, ProceduralGraph
from hpg.tools import World
from hpg.tracing import Tracer


class Engine:
    def __init__(
        self,
        graph: ProceduralGraph,
        world: World,
        router: Router,
        llm: LLM,
        max_steps: int = 20,
        max_tool_rounds: int = 4,
    ):
        self.g = graph
        self.world = world
        self.router = router
        self.llm = llm
        self.max_steps = max_steps
        self.max_tool_rounds = max_tool_rounds

    # ------------------------------------------------------------------ node execution
    def _exec_tool(self, node: Node, state: AgentState) -> dict[str, Any]:
        w, f = self.world, state.facts
        if node.tool == "extract_order_id":
            oid = w.extract_order_id(state.ticket_text)
            f["order_id_in_ticket"] = oid
            return {"order_id": oid}
        if node.tool == "lookup_order":
            r = w.lookup_order(f.get("order_id_in_ticket"))
            state.tool_results[node.id] = r
            f["order_found"] = r["found"]
            if r["found"]:
                f["order_status"] = r["order"]["status"]
            return {"found": r["found"], "status": f.get("order_status")}
        if node.tool in ("check_refund_policy", "issue_refund") and not (
            state.tool_results.get("lookup_order") or {}
        ).get("order"):
            # Reached only after a misroute (router said "found" when it was not): record, don't crash.
            f["tool_error"] = f"{node.tool}: no order available"
            return {"error": f["tool_error"]}
        if node.tool == "check_refund_policy":
            order = state.tool_results["lookup_order"]["order"]
            r = w.check_refund_policy(order)
            state.tool_results[node.id] = r
            for k in ("days_since_delivery", "within_30_day_window", "already_refunded", "duplicate_charge"):
                f[k] = r[k]
            return r
        if node.tool == "issue_refund":
            order = state.tool_results["lookup_order"]["order"]
            if node.id == "refund_duplicate":
                amount = state.tool_results["policy_check"]["duplicate_amount"] or order["amount"]
                r = w.issue_refund(order["order_id"], amount, "duplicate")
            else:
                r = w.issue_refund(order["order_id"], order["amount"], "standard")
            state.ledger = list(w.ledger)
            f["actions_taken"] = _actions_summary(state.ledger)
            return r
        if node.tool == "escalate_to_human":
            reason = f"escalated after {' > '.join(state.path)}"
            r = w.escalate_to_human(reason)
            state.ledger = list(w.ledger)
            f["actions_taken"] = _actions_summary(state.ledger)
            return r
        raise ValueError(f"unknown tool {node.tool!r} on node {node.id}")

    def _exec_llm(self, node: Node, state: AgentState) -> dict[str, Any]:
        if node.llm_task == "investigate_technical":
            return run_investigate_technical(self.llm, self.world, state, self.max_tool_rounds)
        if node.llm_task == "ask_for_info":
            return run_ask_for_info(self.llm, state)
        if node.llm_task == "draft_reply":
            return run_draft_reply(self.llm, state)
        raise ValueError(f"unknown llm_task {node.llm_task!r}")

    # ------------------------------------------------------------------ routing
    def _route(self, node: Node, state: AgentState) -> tuple[Edge, RoutingDecision]:
        outs = self.g.out_edges(node.id)
        if len(outs) == 1:
            return outs[0], RoutingDecision(
                target=outs[0].target, confidence=1.0, probs={outs[0].target: 1.0},
                router="deterministic", latency_ms=0.0,
            )  # fmt: skip
        include_reply = bool((node.laya_questions or {}).get("include_reply"))
        state.routing_extra = state.draft_reply if include_reply else None
        d = self.router.route(state, node.id, outs, node.question or node.description)
        state.routing_extra = None
        edge = next(e for e in outs if e.target == d.target)
        return edge, d

    # ------------------------------------------------------------------ main loop
    def run(self, ticket: dict, tracer: Tracer) -> AgentState:
        self.world.reset()
        state = AgentState(
            ticket_id=ticket["ticket_id"], ticket_text=ticket["text"], current_node=self.g.start
        )
        t_start = time.perf_counter()
        error = None
        node_id = self.g.start
        steps = 0
        try:
            while node_id != END:
                steps += 1
                if steps > self.max_steps:
                    error = f"max_steps ({self.max_steps}) exceeded"
                    break
                node = self.g.node(node_id)
                state.current_node = node_id
                state.path.append(node_id)
                t0 = time.perf_counter()
                if node.executor == "tool":
                    out = self._exec_tool(node, state)
                    tracer.log(type="node", node=node_id, executor="tool", tool=node.tool, output=out,
                               ms=(time.perf_counter() - t0) * 1000)  # fmt: skip
                elif node.executor == "llm":
                    out = self._exec_llm(node, state)
                    tracer.log(type="llm_node", node=node_id, executor="llm", task=node.llm_task,
                               ms=(time.perf_counter() - t0) * 1000, **out)  # fmt: skip
                else:
                    tracer.log(type="node", node=node_id, executor=node.executor, ms=0.0)

                edge, d = self._route(node, state)
                target = edge.target
                note = None
                if edge.max_traversals is not None:
                    key = f"{edge.source}->{edge.target}"
                    used = state.edge_traversals.get(key, 0)
                    if used >= edge.max_traversals:
                        target, note = edge.on_exhausted, f"{key} exhausted after {used} traversal(s)"
                    else:
                        state.edge_traversals[key] = used + 1
                        state.review_feedback = edge.condition
                tracer.log(
                    type="route", node=node_id, path=list(state.path), target=target,
                    chosen_edge=edge.target, note=note, **d.model_dump(exclude={"target"}),
                )  # fmt: skip
                if d.router != "deterministic" and note is None:
                    state.decision_edges.append(edge)
                state.incoming_edge = edge
                node_id = target
        except Exception as e:  # noqa: BLE001 - record and stop; eval counts it as a failure
            error = f"{type(e).__name__}: {e}"
        if node_id == END:
            state.path.append(END)
        tracer.log(
            type="end", path=state.path, ledger=state.ledger, reply=state.draft_reply,
            info_request=state.info_request, error=error, wall_ms=(time.perf_counter() - t_start) * 1000,
        )  # fmt: skip
        return state


def _actions_summary(ledger: list[dict]) -> str:
    parts = []
    for e in ledger:
        if e["action"] == "refund":
            parts.append(f"refund {e['amount']} ({e['reason']})")
        else:
            parts.append("escalated to human")
    return ", ".join(parts) or "none"
