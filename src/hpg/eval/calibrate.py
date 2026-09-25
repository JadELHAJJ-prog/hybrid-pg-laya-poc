"""Phase 5: collect routing decisions on dev, calibrate Laya, choose checkpoint, gate and tau (dev only).

1. collect():  walk every dev ticket along its GOLD path (teacher forcing) with the real LLM nodes, and at each
               multi-way decision record the LLMRouter decision plus a snapshot of the exact routing input.
2. score_laya(): replay the snapshots through a Laya checkpoint with temperatures neutralised (T=1) → raw probs.
3. analyze():  accuracy, ECE (raw / shipped-T / fitted-T with k-fold CV), reliability diagrams,
               accuracy-vs-coverage, simulated hybrid accuracy vs tau → choose tau.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from hpg.routers.base import Router, RoutingDecision
from hpg.schema import AgentState

ROUTING_NODES = ("guard", "classify", "lookup_order", "policy_check", "lookup_order_delivery", "final_review")


# ----------------------------------------------------------------------------- 1. collect
class ProbeRouter(Router):
    """Follows the gold path; asks `inner` (the LLM router) at every decision and records everything."""

    name = "probe"

    def __init__(self, inner: Router, gold: list[str], sink: list[dict], ticket: dict):
        self.inner, self.gold, self.sink, self.ticket = inner, gold, sink, ticket

    def route(self, state: AgentState, node_id: str, edges, question: str) -> RoutingDecision:
        gold_t = self.gold[len(state.path)]
        d = self.inner.route(state, node_id, edges, question)
        self.sink.append(
            {
                "ticket_id": state.ticket_id,
                "tags": self.ticket.get("tags", []),
                "node": node_id,
                "question": question,
                "targets": [e.target for e in edges],
                "gold_target": gold_t,
                "snapshot": {
                    "ticket_text": state.ticket_text,
                    "facts": state.facts,
                    "routing_extra": state.routing_extra,
                    "path": list(state.path),
                },
                "llm": d.model_dump(),
                "llm_correct": d.target == gold_t,
            }
        )
        return d.model_copy(update={"target": gold_t})


def collect(cfg: dict, root: Path, split: str = "dev", limit: int | None = None) -> Path:
    from hpg.cli import build, load_tickets
    from hpg.tracing import Tracer

    out = root / cfg["paths"]["runs"] / f"calib_{split}"
    out.mkdir(parents=True, exist_ok=True)
    eng = build(cfg, "llm")
    llm_router = eng.router
    f = out / "decisions.jsonl"
    done = set()
    if f.exists():
        done = {json.loads(x)["ticket_id"] for x in f.read_text().splitlines() if x}
    tickets = [t for t in load_tickets(cfg) if t["split"] == split][:limit]
    for i, t in enumerate(tickets, 1):
        if t["ticket_id"] in done:
            continue
        sink: list[dict] = []
        eng.router = ProbeRouter(llm_router, t["gold_path"], sink, t)
        eng.run(t, Tracer(out / "traces" / f"{t['ticket_id']}.jsonl"))
        with f.open("a") as fh:
            for r in sink:
                fh.write(json.dumps(r, default=str) + "\n")
        if i % 10 == 0:
            print(f"  collect {split}: {i}/{len(tickets)}", flush=True)
    return f


# ----------------------------------------------------------------------------- 2. replay through Laya
def score_laya(cfg: dict, root: Path, name: str, subfolder: str | None, split: str = "dev",
               guard_preset: bool = True, path: str | None = None, device: str | None = None) -> Path:  # fmt: skip
    from hpg.graph_loader import load_graph
    from hpg.routers.laya_router import LayaRouter, load_agent

    calib = root / cfg["paths"]["runs"] / f"calib_{split}"
    rows = [json.loads(x) for x in (calib / "decisions.jsonl").read_text().splitlines() if x]
    graph = load_graph(root / cfg["paths"]["graph"])
    agent = load_agent(path or cfg["laya"]["repo"], None if path else subfolder, device or cfg["laya"]["device"],
                       cfg["laya"].get("bf16_weights", True))  # fmt: skip
    shipped = dict(agent.temperature_by_options)
    for b in ("2", "3-5", "6-10", "11+"):  # neutralise choice temperatures -> raw softmax(logits)
        agent.temperature_by_options[f"choice:{b}"] = 1.0
    r_choice = LayaRouter(agent, graph, guard_mode="choice")
    r_preset = LayaRouter(agent, graph, guard_mode="preset")
    r_choice.warmup()
    out = calib / f"laya_{name}.jsonl"
    with out.open("w") as fh:
        for row in rows:
            s = row["snapshot"]
            st = AgentState(ticket_id=row["ticket_id"], ticket_text=s["ticket_text"], current_node=row["node"],
                            path=s["path"], facts=s["facts"], routing_extra=s["routing_extra"])  # fmt: skip
            edges = graph.out_edges(row["node"])
            d = r_choice.route(st, row["node"], edges, row["question"])
            rec = {"ticket_id": row["ticket_id"], "node": row["node"], "gold_target": row["gold_target"],
                   "targets": row["targets"], "tags": row["tags"], "laya": d.model_dump(),
                   "llm_correct": row["llm_correct"], "llm_target": row["llm"]["target"],
                   "llm_latency_ms": row["llm"]["latency_ms"],
                   "shipped_T": shipped.get(f"choice:{_bucket(len(edges) + 1)}")}  # fmt: skip
            if row["node"] == "guard" and guard_preset:
                rec["guard_preset"] = r_preset.route(st, row["node"], edges, row["question"]).model_dump()
            fh.write(json.dumps(rec, default=str) + "\n")
    del agent
    import torch

    torch.cuda.empty_cache()
    return out


def _bucket(n: int) -> str:
    return "2" if n == 2 else "3-5" if n <= 5 else "6-10" if n <= 10 else "11+"


# ----------------------------------------------------------------------------- 3. analysis helpers
def option_probs(rec: dict) -> tuple[list[str], np.ndarray, int]:
    """Probabilities over [edge targets..., NONE] in edge order, and index of the gold option."""
    targets = rec["targets"]
    p = rec["laya"]["probs"]
    arr = np.array([p.get(t, 0.0) for t in targets] + [p.get("NONE", 0.0)], dtype=float)
    arr = np.clip(arr, 1e-6, None)
    arr /= arr.sum()
    return targets, arr, targets.index(rec["gold_target"])


def apply_T(p: np.ndarray, T: float) -> np.ndarray:
    z = np.log(p) / T
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def decide(targets: list[str], p: np.ndarray) -> tuple[str, bool, float, float]:
    """Router semantics: argmax; if NONE wins, take best real edge but flag it. Returns
    (target, none_won, answer_confidence, entropy_confidence)."""
    k = len(p)
    none_won = int(np.argmax(p)) == k - 1
    best_real = int(np.argmax(p[:-1]))
    ent = -float((p * np.log(p)).sum()) / math.log(k)
    return targets[best_real], none_won, float(p.max()), 1.0 - ent


def fit_T(ps: list[np.ndarray], gold: list[int]) -> float:
    """Temperature minimising NLL of the gold option (grid + refine), clamped like laya to [0.5, 5]."""
    if not ps:
        return 1.0

    def nll(T: float) -> float:
        return -float(np.mean([np.log(apply_T(p, T)[g]) for p, g in zip(ps, gold, strict=True)]))

    grid = np.exp(np.linspace(np.log(0.05), np.log(5.0), 120))
    best = min(grid, key=nll)
    fine = np.linspace(best * 0.85, best * 1.15, 40)
    # Clamp like laya's own loader ([0.5, 5]): with perfectly separable dev decisions NLL drives T -> 0.
    return float(np.clip(min(fine, key=nll), 0.5, 5.0))


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def reliability(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> list[tuple[float, float, int]]:
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            out.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return out


def evaluate_records(recs: list[dict], temps: dict[int, float] | None) -> dict[str, np.ndarray]:
    tgt, nonew, ac, ec, corr, llmc, n = [], [], [], [], [], [], []
    for r in recs:
        targets, p, g = option_probs(r)
        if temps is not None:
            p = apply_T(p, temps.get(len(p), 1.0))
        t, none_won, a, e = decide(targets, p)
        tgt.append(t)
        nonew.append(none_won)
        ac.append(a)
        ec.append(e)
        corr.append(t == r["gold_target"])
        llmc.append(bool(r["llm_correct"]))
        n.append(len(p))
    return {k: np.array(v) for k, v in dict(target=tgt, none_won=nonew, answer_conf=ac, entropy_conf=ec,
                                            correct=corr, llm_correct=llmc, n=n).items()}  # fmt: skip


def cv_temps(recs: list[dict], k: int = 5, seed: int = 0) -> tuple[dict[str, np.ndarray], dict[int, float]]:
    """k-fold CV (grouped by ticket) calibrated predictions + temperatures fitted on all dev (for deployment)."""
    rng = np.random.default_rng(seed)
    tids = sorted({r["ticket_id"] for r in recs})
    rng.shuffle(tids)
    fold = {t: i % k for i, t in enumerate(tids)}

    def fit(sub: list[dict]) -> dict[int, float]:
        by_n: dict[int, tuple[list, list]] = {}
        for r in sub:
            _, p, g = option_probs(r)
            by_n.setdefault(len(p), ([], []))
            by_n[len(p)][0].append(p)
            by_n[len(p)][1].append(g)
        return {n: fit_T(ps, gs) for n, (ps, gs) in by_n.items()}

    parts: list[tuple[int, dict]] = []
    for i in range(k):
        train = [r for r in recs if fold[r["ticket_id"]] != i]
        test_idx = [j for j, r in enumerate(recs) if fold[r["ticket_id"]] == i]
        ev = evaluate_records([recs[j] for j in test_idx], fit(train))
        parts.append((i, {"idx": np.array(test_idx), **ev}))
    merged: dict[str, np.ndarray] = {}
    order = np.concatenate([p["idx"] for _, p in parts])
    inv = np.argsort(order)
    for key in parts[0][1]:
        if key == "idx":
            continue
        merged[key] = np.concatenate([p[key] for _, p in parts])[inv]
    return merged, fit(recs)


def hybrid_curve(ev: dict[str, np.ndarray], gate: str, taus: np.ndarray) -> list[dict[str, float]]:
    score = ev["answer_conf"] if gate == "answer_confidence" else ev["entropy_conf"]
    rows = []
    for tau in taus:
        use_laya = (score >= tau) & ~ev["none_won"]
        acc = np.where(use_laya, ev["correct"], ev["llm_correct"]).mean()
        cov = use_laya.mean()
        laya_acc_covered = ev["correct"][use_laya].mean() if use_laya.any() else float("nan")
        rows.append({"tau": float(tau), "hybrid_acc": float(acc), "coverage": float(cov),
                     "fallback_rate": float(1 - cov), "laya_acc_on_covered": float(laya_acc_covered)})  # fmt: skip
    return rows


def choose_tau(curve: list[dict], llm_acc: float, margin: float = 0.02) -> dict[str, Any]:
    ok = [r for r in curve if r["hybrid_acc"] >= llm_acc - margin]
    return min(ok, key=lambda r: r["tau"]) if ok else max(curve, key=lambda r: r["hybrid_acc"])
