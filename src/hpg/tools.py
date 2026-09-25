"""Deterministic mock tools over the RefundDesk world. Every call is recorded for evaluation."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

ORDER_RE = re.compile(r"ORD-\d{5}")
REFUND_WINDOW_DAYS = 30
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class World:
    """Loaded mock data plus a per-ticket action ledger and call log."""

    def __init__(self, orders_path: Path, kb_path: Path, status_path: Path, today: str):
        raw = json.loads(Path(orders_path).read_text())
        self.orders = {o["order_id"]: {k: v for k, v in o.items() if not k.startswith("_")} for o in raw}
        self.kb = json.loads(Path(kb_path).read_text())
        self.status = json.loads(Path(status_path).read_text())
        self.today = date.fromisoformat(today)
        self._bm25 = BM25Okapi([_tok(f"{a['title']} {a['keywords']} {a['body']}") for a in self.kb])
        self.reset()

    @classmethod
    def from_config(cls, cfg: dict, root: Path | None = None) -> World:
        root = root or Path.cwd()
        p = cfg["paths"]
        return cls(root / p["orders"], root / p["kb"], root / p["service_status"], cfg["today"])

    def reset(self) -> None:
        self.ledger: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []

    def _record(self, tool: str, args: dict, result: Any) -> Any:
        self.calls.append({"tool": tool, "args": args, "result": result})
        return result

    # ---- tools ---------------------------------------------------------------
    def extract_order_id(self, text: str) -> str | None:
        m = ORDER_RE.search(text)
        return self._record("extract_order_id", {"text": text[:80]}, m.group(0) if m else None)

    def lookup_order(self, order_id: str | None) -> dict:
        order = self.orders.get(order_id) if order_id else None
        res = {"found": order is not None, "order_id": order_id, "order": order}
        return self._record("lookup_order", {"order_id": order_id}, res)

    def check_refund_policy(self, order: dict) -> dict:
        delivered = order.get("delivered_date")
        days = (self.today - date.fromisoformat(delivered)).days if delivered else None
        amounts = [c["amount"] for c in order.get("charges", [])]
        dup = len(amounts) != len(set(amounts))
        res = {
            "order_status": order["status"],
            "days_since_delivery": days,
            "within_30_day_window": days is not None and days <= REFUND_WINDOW_DAYS,
            "already_refunded": bool(order.get("refunded")),
            "duplicate_charge": dup,
            "duplicate_amount": amounts[0] if dup else None,
        }
        return self._record("check_refund_policy", {"order_id": order["order_id"]}, res)

    def issue_refund(self, order_id: str, amount: float, reason: str) -> dict:
        entry = {
            "action": "refund",
            "order_id": order_id,
            "amount": round(float(amount), 2),
            "reason": reason,
        }
        self.ledger.append(entry)
        return self._record("issue_refund", {"order_id": order_id, "amount": amount, "reason": reason}, entry)

    def search_kb(self, query: str, k: int = 3) -> list[dict]:
        scores = self._bm25.get_scores(_tok(query))
        top = sorted(range(len(self.kb)), key=lambda i: -scores[i])[:k]
        res = [
            {"id": self.kb[i]["id"], "title": self.kb[i]["title"], "body": self.kb[i]["body"]}
            for i in top
            if scores[i] > 0
        ]
        return self._record("search_kb", {"query": query, "k": k}, res)

    def check_service_status(self, service: str) -> dict:
        s = self.status.get(service)
        res = {"service": service, "known": s is not None, **(s or {})}
        return self._record("check_service_status", {"service": service}, res)

    def escalate_to_human(self, reason: str) -> dict:
        entry = {"action": "escalate", "reason": reason}
        self.ledger.append(entry)
        return self._record("escalate_to_human", {"reason": reason}, entry)
