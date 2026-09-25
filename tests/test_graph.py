import pytest

from hpg.graph_loader import GraphValidationError, is_valid_path, validate
from hpg.schema import END, Edge, Node, ProceduralGraph


def test_refunddesk_loads(graph):
    assert graph.start == "intake"
    assert len(graph.out_edges("classify")) == 4
    assert len(graph.out_edges("policy_check")) == 3
    retry = graph.edge("final_review", "draft_reply")
    assert retry.max_traversals == 1 and retry.on_exhausted == "escalate_human"


def test_valid_paths(graph):
    ok = ["intake", "guard", "classify", "lookup_order", "policy_check", "refund_duplicate",
          "draft_reply", "final_review", "END"]  # fmt: skip
    assert is_valid_path(graph, ok)
    assert is_valid_path(graph, ["intake", "guard", "escalate_human", "END"])
    assert is_valid_path(graph, ok[:-1] + ["draft_reply", "final_review", "escalate_human", "END"])
    assert not is_valid_path(graph, ["intake", "classify", "END"])
    assert not is_valid_path(graph, ["guard", "escalate_human", "END"])


def _g(nodes, edges, start="a"):
    mk = lambda s, t, **kw: Edge(source=s, target=t, condition="c", guidance="g", pitfalls="p", **kw)  # noqa: E731
    return ProceduralGraph(
        nodes=[Node(id=n, description=n, executor="terminal" if n == END else "route") for n in nodes],
        edges=[mk(*e[:2], **(e[2] if len(e) > 2 else {})) for e in edges],
        start=start,
    )


def test_rejects_unknown_endpoint():
    with pytest.raises(GraphValidationError, match="unknown node"):
        validate(_g(["a", END], [("a", "b"), ("a", END)]))


def test_rejects_dead_end():
    with pytest.raises(GraphValidationError, match="no outgoing"):
        validate(_g(["a", "b", END], [("a", "b"), ("a", END)]))


def test_rejects_undeclared_cycle():
    with pytest.raises(GraphValidationError, match="cycle"):
        validate(_g(["a", "b", END], [("a", "b"), ("b", "a"), ("b", END)]))


def test_accepts_declared_cycle():
    validate(
        _g(["a", "b", END], [("a", "b"), ("b", "a", {"max_traversals": 1, "on_exhausted": END}), ("b", END)])
    )


def test_rejects_unreachable():
    with pytest.raises(GraphValidationError, match="unreachable"):
        validate(_g(["a", "b", END], [("a", END), ("b", END)]))
