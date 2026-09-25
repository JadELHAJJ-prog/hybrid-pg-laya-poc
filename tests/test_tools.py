import json

from conftest import ROOT

RAW = {o["order_id"]: o for o in json.loads((ROOT / "data/orders.json").read_text())}


def first(bucket):
    return next(o["order_id"] for o in RAW.values() if o["_bucket"] == bucket)


def test_extract_order_id(world):
    assert world.extract_order_id("hi, ORD-10231 was charged twice") == "ORD-10231"
    assert world.extract_order_id("ord-10231 lowercase") is None
    assert world.extract_order_id("no id here") is None
    assert world.extract_order_id("ORD-1234 too short") is None
    assert world.extract_order_id("first ORD-10001 then ORD-10002") == "ORD-10001"


def test_lookup_order(world):
    r = world.lookup_order(first("refundable"))
    assert r["found"] and r["order"]["status"] == "delivered"
    assert "_bucket" not in r["order"]
    assert world.lookup_order("ORD-99999") == {"found": False, "order_id": "ORD-99999", "order": None}
    assert world.lookup_order(None)["found"] is False


def test_refund_policy_buckets(world):
    def facts(bucket):
        return world.check_refund_policy(world.lookup_order(first(bucket))["order"])

    f = facts("refundable")
    assert f["within_30_day_window"] and not f["already_refunded"] and not f["duplicate_charge"]
    assert 0 < f["days_since_delivery"] <= 30
    f = facts("outside_window")
    assert not f["within_30_day_window"] and f["days_since_delivery"] > 30
    assert facts("already_refunded")["already_refunded"]
    f = facts("duplicate")
    assert f["duplicate_charge"] and f["duplicate_amount"] > 0
    assert facts("duplicate_old")["duplicate_charge"] and not facts("duplicate_old")["within_30_day_window"]
    f = facts("processing")
    assert f["days_since_delivery"] is None and not f["within_30_day_window"]


def test_ledger_and_calls(world):
    world.issue_refund("ORD-10001", 79.99, "standard")
    world.escalate_to_human("lost package")
    assert [e["action"] for e in world.ledger] == ["refund", "escalate"]
    assert [c["tool"] for c in world.calls] == ["issue_refund", "escalate_to_human"]
    world.reset()
    assert world.ledger == [] and world.calls == []


def test_search_kb(world):
    res = world.search_kb("app crashes when I log in on android")
    assert res and res[0]["id"] == "KB02"
    assert world.search_kb("password reset email")[0]["id"] == "KB01"
    assert world.search_kb("zzzz qqqq") == []


def test_service_status(world):
    assert world.check_service_status("login")["status"] == "degraded"
    assert world.check_service_status("shipping-api")["status"] == "operational"
    assert world.check_service_status("nope")["known"] is False
