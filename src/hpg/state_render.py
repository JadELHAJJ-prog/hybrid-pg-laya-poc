"""Compact, token-budgeted state rendering for Laya (plan §5.1).

Only the ticket text and short key facts are rendered. When over budget, the ticket is truncated
(head + tail kept, middle elided); facts are never truncated.
"""

from __future__ import annotations

from collections.abc import Callable

from hpg.schema import AgentState

FACT_ORDER = [
    "order_id_in_ticket",
    "order_found",
    "order_status",
    "days_since_delivery",
    "within_30_day_window",
    "already_refunded",
    "duplicate_charge",
    "actions_taken",
]


def render_facts(facts: dict) -> str:
    keys = [k for k in FACT_ORDER if k in facts] + sorted(k for k in facts if k not in FACT_ORDER)
    parts = []
    for k in keys:
        v = facts[k]
        if isinstance(v, bool):
            v = str(v).lower()
        parts.append(f"{k}: {v if v is not None else 'none'}")
    return "; ".join(parts)


def _clip_words(words: list[str], keep: int) -> str:
    if keep >= len(words):
        return " ".join(words)
    head = max(1, int(keep * 0.6))
    tail = max(1, keep - head)
    return " ".join(words[:head]) + " [...] " + " ".join(words[-tail:])


def render_state_for_laya(
    state: AgentState,
    max_tokens: int,
    count_tokens: Callable[[dict], int],
    extra: str | None = None,
) -> dict[str, str]:
    """Return a Laya state dict {"ticket": ..., "facts": ..., ["reply": ...]} within `max_tokens`.

    `count_tokens(state_dict)` must count tokens the way Laya does for that dict (json.dumps + tokenizer).
    `extra` (e.g. a draft reply for final_review) is kept whole
    like facts.
    """
    facts = render_facts(state.facts)
    fixed = {"facts": facts} if facts else {}
    if extra:
        fixed["reply"] = extra

    def total(ticket: str) -> int:
        return count_tokens({"ticket": ticket, **fixed})

    words = state.ticket_text.split()
    ticket = state.ticket_text
    if total(ticket) > max_tokens:
        lo, hi = 0, len(words)
        while lo < hi:  # largest word budget that fits
            mid = (lo + hi + 1) // 2
            if total(_clip_words(words, mid)) <= max_tokens:
                lo = mid
            else:
                hi = mid - 1
        ticket = _clip_words(words, lo)
    return {"ticket": ticket, **fixed}
