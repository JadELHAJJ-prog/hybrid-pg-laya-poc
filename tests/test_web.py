"""Web UI endpoints that don't need models (results dashboard, graph, examples, input validation)."""

from fastapi.testclient import TestClient

from hpg.web.server import app

client = TestClient(app)


def test_index_and_static():
    r = client.get("/")
    assert r.status_code == 200 and "Procedural Graph" in r.text


def test_graph_endpoint(graph):
    g = client.get("/api/graph").json()
    assert {n["id"] for n in g["nodes"]} == {n.id for n in graph.nodes}
    assert any(e["retry"] for e in g["edges"])


def test_examples_and_status():
    ex = client.get("/api/examples").json()
    assert len(ex["examples"]) >= 6 and len(ex["orders"]) == 60
    st = client.get("/api/status").json()
    assert "hybrid_ft" in st["modes"]


def test_results_endpoint_matches_reports():
    import json
    from pathlib import Path

    r = client.get("/api/results").json()
    summary = json.loads(Path("reports/summary_test.json").read_text())
    for e, s in summary.items():
        assert r["experiments"][e]["routing_acc"] == s["routing_acc"]
        assert r["failures"][e]["total"] == round((1 - s["final_action_acc"]) * s["n_tickets"])


def test_chat_rejects_bad_input():
    assert client.post("/api/chat", json={"text": "  ", "mode": "hybrid_ft"}).status_code == 400
    assert client.post("/api/chat", json={"text": "hi", "mode": "nope"}).status_code == 400
    assert client.get("/reports/../config.yaml").status_code == 404
