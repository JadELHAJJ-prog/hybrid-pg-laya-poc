"""Metrics computed from traces only (plan §9)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hpg.tracing import read_trace

ROUTING_NODES = ("classify", "lookup_order", "policy_check", "lookup_order_delivery")
CHECK_NODES = ("guard", "final_review")


def load_tickets(path: Path) -> dict[str, dict]:
    return {t["ticket_id"]: t for t in (json.loads(x) for x in Path(path).read_text().splitlines() if x)}


def final_action_from_trace(steps: list[dict]) -> str:
    """Derive the final action from the ledger + path in the trace (same vocabulary as gold)."""
    end = next((s for s in reversed(steps) if s.get("type") == "end"), None)
    ledger = end["ledger"] if end else []
    path = end["path"] if end else [s["node"] for s in steps if s.get("type") == "node"]
    if any(e["action"] == "escalate" for e in ledger):
        return "escalate_human"
    refunds = [e for e in ledger if e["action"] == "refund"]
    if refunds:
        return "refund_duplicate" if refunds[-1]["reason"] == "duplicate" else "refund_standard"
    for node, action in (
        ("deny_with_alternative", "deny_with_alternative"),
        ("ask_for_info", "ask_for_info"),
        ("investigate_technical", "technical_reply"),
        ("lookup_order_delivery", "delivery_update"),
    ):
        if node in path:
            return action
    return "none"


def decisions_frame(run_dir: Path, tickets: dict[str, dict]) -> pd.DataFrame:
    """One row per multi-way decision the engine made, scored against gold where on the gold prefix."""
    rows = []
    for f in sorted(Path(run_dir).glob("T*.jsonl")):
        steps = read_trace(f)
        t = tickets[f.stem]
        gold = t["gold_path"]
        for s in steps:
            if s.get("type") != "route":
                continue
            node, target = s["node"], s["target"]
            walked = s["path"]  # path so far, ending at `node`
            k = len(walked)
            gold_next = gold[k] if gold[:k] == walked and k < len(gold) else None
            rows.append(
                {
                    "ticket_id": f.stem,
                    "split": t["split"],
                    "node": node,
                    "router": s["router"],
                    "target": target,
                    "gold_target": gold_next,
                    "scorable": gold_next is not None,
                    "correct": gold_next is not None and target == gold_next,
                    "confidence": s.get("confidence"),
                    "answer_confidence": s.get("answer_confidence"),
                    "laya_confidence": s.get("laya_confidence"),
                    "laya_target": s.get("laya_target"),
                    "latency_ms": s["latency_ms"],
                    "llm_tokens": s.get("llm_prompt_tokens", 0) + s.get("llm_completion_tokens", 0),
                    "n_options": len(s.get("probs") or {}),
                    "tags": ",".join(t.get("tags", [])),
                }
            )
    return pd.DataFrame(rows)


def ticket_frame(run_dir: Path, tickets: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(run_dir).glob("T*.jsonl")):
        steps = read_trace(f)
        t = tickets[f.stem]
        end = next((s for s in reversed(steps) if s.get("type") == "end"), {})
        path = end.get("path", [])
        routes = [s for s in steps if s.get("type") == "route"]
        llm_nodes = [s for s in steps if s.get("type") == "llm_node"]
        route_llm = [s for s in routes if s["router"] in ("llm", "llm_fallback")]
        rows.append(
            {
                "ticket_id": f.stem,
                "split": t["split"],
                "category": t["gold_category"],
                "tags": ",".join(t.get("tags", [])),
                "path_exact": path == t["gold_path"],
                "final_action": final_action_from_trace(steps),
                "gold_final_action": t["gold_final_action"],
                "action_correct": final_action_from_trace(steps) == t["gold_final_action"],
                "e2e_ms": end.get("wall_ms", np.nan),
                "llm_calls_routing": len(route_llm),
                "llm_calls_nodes": sum(s.get("llm_calls", 1) for s in llm_nodes),
                "llm_tokens_routing": sum(
                    s.get("llm_prompt_tokens", 0) + s.get("llm_completion_tokens", 0) for s in route_llm
                ),
                "llm_tokens_nodes": sum(
                    s.get("llm_prompt_tokens", 0) + s.get("llm_completion_tokens", 0) for s in llm_nodes
                ),
                "fallbacks": sum(s["router"] == "llm_fallback" for s in routes),
                "laya_routes": sum(s["router"] == "laya" for s in routes),
                "guard_flagged": "guard" in path and path[path.index("guard") + 1] == "escalate_human",
                "review_pass_first": _review_first_pass(steps),
                "error": end.get("error"),
            }
        )
    df = pd.DataFrame(rows)
    df["llm_calls"] = df["llm_calls_routing"] + df["llm_calls_nodes"]
    df["llm_tokens"] = df["llm_tokens_routing"] + df["llm_tokens_nodes"]
    return df


def _review_first_pass(steps: list[dict]) -> bool | None:
    for s in steps:
        if s.get("type") == "route" and s["node"] == "final_review":
            return s["target"] == "END"
    return None


def pct(x: pd.Series, q: float) -> float:
    x = x.dropna()
    return float(np.percentile(x, q)) if len(x) else float("nan")


def summarize(run_dir: Path, tickets: dict[str, dict]) -> dict[str, Any]:
    d = decisions_frame(run_dir, tickets)
    t = ticket_frame(run_dir, tickets)
    rd = d[d.node.isin(ROUTING_NODES) & d.scorable]
    guard = d[(d.node == "guard") & d.scorable]
    adv = t.tags.str.contains("adversarial")
    out: dict[str, Any] = {
        "n_tickets": len(t),
        "routing_acc": float(rd.correct.mean()) if len(rd) else float("nan"),
        "routing_n": int(len(rd)),
        "decision_acc_incl_guard": float(pd.concat([rd, guard]).correct.mean()) if len(rd) else float("nan"),
        "path_exact": float(t.path_exact.mean()),
        "final_action_acc": float(t.action_correct.mean()),
        "guard_recall_adv": float(t[adv].guard_flagged.mean()) if adv.any() else float("nan"),
        "guard_fpr_clean": float(t[~adv].guard_flagged.mean()) if (~adv).any() else float("nan"),
        "route_p50_ms": pct(d[d.node.isin(ROUTING_NODES)].latency_ms, 50),
        "route_p95_ms": pct(d[d.node.isin(ROUTING_NODES)].latency_ms, 95),
        "e2e_p50_ms": pct(t.e2e_ms, 50),
        "e2e_p95_ms": pct(t.e2e_ms, 95),
        "llm_calls_per_ticket": float(t.llm_calls.mean()),
        "llm_calls_routing_per_ticket": float(t.llm_calls_routing.mean()),
        "llm_tokens_per_ticket": float(t.llm_tokens.mean()),
        "llm_tokens_routing_per_ticket": float(t.llm_tokens_routing.mean()),
        "llm_tokens_nodes_per_ticket": float(t.llm_tokens_nodes.mean()),
        "fallback_rate": float((d[d.node.isin(ROUTING_NODES + CHECK_NODES)].router == "llm_fallback").mean())
        if len(d)
        else float("nan"),
        "review_first_pass_rate": float(t.review_pass_first.dropna().mean())
        if t.review_pass_first.notna().any()
        else float("nan"),
        "errors": int(t.error.notna().sum()),
    }
    by_router = (
        d[d.node.isin(ROUTING_NODES)]
        .groupby("router")
        .latency_ms.agg(p50=lambda x: pct(x, 50), p95=lambda x: pct(x, 95), n="count")
    )
    out["route_latency_by_router"] = by_router.round(1).to_dict(orient="index")
    out["routing_acc_by_node"] = rd.groupby("node").correct.mean().round(3).to_dict()
    return out
