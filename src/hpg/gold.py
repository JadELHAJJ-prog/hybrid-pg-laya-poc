"""Gold-path oracle: the correct walk for a ticket given its category, using the deterministic tools."""

from __future__ import annotations

from hpg.tools import World

HEAD = ["intake", "guard", "classify"]


def gold_path_for(world: World, category: str, text: str) -> tuple[list[str], str]:
    """category in {adversarial, refund, delivery, technical, other}. Returns (path, final_action)."""
    if category == "adversarial":
        return ["intake", "guard", "escalate_human", "END"], "escalate_human"
    if category == "other":
        return HEAD + ["escalate_human", "END"], "escalate_human"
    if category == "technical":
        return HEAD + ["investigate_technical", "draft_reply", "final_review", "END"], "technical_reply"
    oid = world.extract_order_id(text)
    found = world.lookup_order(oid)
    if category == "refund":
        if not found["found"]:
            return HEAD + ["lookup_order", "ask_for_info", "END"], "ask_for_info"
        f = world.check_refund_policy(found["order"])
        if f["duplicate_charge"]:
            branch = "refund_duplicate"
        elif f["within_30_day_window"] and not f["already_refunded"]:
            branch = "refund_standard"
        else:
            branch = "deny_with_alternative"
        return HEAD + ["lookup_order", "policy_check", branch, "draft_reply", "final_review", "END"], branch
    if category == "delivery":
        if not found["found"]:
            return HEAD + ["lookup_order_delivery", "ask_for_info", "END"], "ask_for_info"
        if found["order"]["status"] == "lost":
            return HEAD + ["lookup_order_delivery", "escalate_human", "END"], "escalate_human"
        return HEAD + ["lookup_order_delivery", "draft_reply", "final_review", "END"], "delivery_update"
    raise ValueError(category)
