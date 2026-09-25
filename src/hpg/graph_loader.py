"""Load a Procedural Graph from YAML and validate its structure with networkx."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import yaml

from hpg.schema import END, ProceduralGraph


class GraphValidationError(ValueError):
    pass


def to_networkx(g: ProceduralGraph) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for n in g.nodes:
        G.add_node(n.id, executor=n.executor)
    for e in g.edges:
        G.add_edge(e.source, e.target, condition=e.condition)
        if e.on_exhausted:
            G.add_edge(e.source, e.on_exhausted, condition="(on_exhausted)")
    return G


def validate(g: ProceduralGraph) -> None:
    ids = [n.id for n in g.nodes]
    errors: list[str] = []
    if len(ids) != len(set(ids)):
        errors.append("duplicate node ids")
    idset = set(ids)
    if END not in idset:
        errors.append("graph has no END node")
    if g.start not in idset:
        errors.append(f"start node {g.start!r} missing")
    for e in g.edges:
        for end in (e.source, e.target, e.on_exhausted):
            if end is not None and end not in idset:
                errors.append(f"edge {e.source}->{e.target} references unknown node {end!r}")
        if (e.max_traversals is None) != (e.on_exhausted is None):
            errors.append(f"edge {e.source}->{e.target}: max_traversals and on_exhausted go together")
        if not (e.condition.strip() and e.guidance.strip() and e.pitfalls.strip()):
            errors.append(f"edge {e.source}->{e.target} missing condition/guidance/pitfalls text")
    pairs = [(e.source, e.target) for e in g.edges]
    if len(pairs) != len(set(pairs)):
        errors.append("duplicate edges")
    for n in g.nodes:
        outs = g.out_edges(n.id)
        if n.id == END and outs:
            errors.append("END must have no outgoing edges")
        if n.id != END and not outs:
            errors.append(f"non-terminal node {n.id!r} has no outgoing edges")
        if n.executor == "tool" and not n.tool:
            errors.append(f"tool node {n.id!r} has no tool")
        if n.executor == "llm" and not n.llm_task:
            errors.append(f"llm node {n.id!r} has no llm_task")
    if errors:
        raise GraphValidationError("; ".join(errors))

    G = to_networkx(g)
    unreachable = idset - set(nx.descendants(G, g.start)) - {g.start}
    if unreachable:
        raise GraphValidationError(f"nodes unreachable from start: {sorted(unreachable)}")
    cant_finish = [n for n in idset if n != END and not nx.has_path(G, n, END)]
    if cant_finish:
        raise GraphValidationError(f"nodes that cannot reach END: {sorted(cant_finish)}")
    # Cycles are only allowed through declared bounded back-edges.
    H = nx.DiGraph()
    H.add_nodes_from(idset)
    H.add_edges_from((e.source, e.target) for e in g.edges if e.max_traversals is None)
    if not nx.is_directed_acyclic_graph(H):
        raise GraphValidationError(f"undeclared cycle(s): {list(nx.simple_cycles(H))[:3]}")


def load_graph(path: str | Path) -> ProceduralGraph:
    data = yaml.safe_load(Path(path).read_text())
    g = ProceduralGraph.model_validate(data)
    validate(g)
    return g


def is_valid_path(g: ProceduralGraph, path: list[str]) -> bool:
    """True if `path` is a start->END walk over declared edges (on_exhausted counts as an edge)."""
    if not path or path[0] != g.start or path[-1] != END:
        return False
    for a, b in zip(path, path[1:], strict=False):
        outs = g.out_edges(a)
        if not any(e.target == b or e.on_exhausted == b for e in outs):
            return False
    return True
