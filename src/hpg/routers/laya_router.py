"""System 1 router: Laya answers one opaque-key `choice` question per decision node (plan §5.2.2, §5.5).

Verified against installed laya 0.3.20 (see RESULTS.md Phase 1):
- `laya.load(repo, subfolder=..., device=...)` -> Agent; `agent.predict(state, questions)`.
- choice answer: {choice, probabilities, confidence (1 - normalised entropy), answer_confidence (max p)}.
- state dicts are serialised with json.dumps; state is truncated from the right past max_len - head_max_len,
  so we pre-truncate the ticket ourselves to keep facts intact.
- temperatures: agent.temperature_by_options["choice:<bucket>"] applied to logits before softmax.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

import numpy as np

from hpg.routers.base import NONE_KEY, NONE_TEXT, Router, RoutingDecision, option_keys
from hpg.schema import AgentState, Edge, ProceduralGraph
from hpg.state_render import render_state_for_laya

# harm_severity levels: 0 none, 1 minor, 2 serious, 3 severe
GUARD_DEFAULTS = {"noul_threshold": 0.5, "harm_threshold": 1.5}


def load_agent(repo: str, subfolder: str | None, device: str, bf16_weights: bool = True):
    """Load a Laya agent. On CUDA, store weights in bf16 (stock keeps fp32 weights + bf16 autocast, ~2.1 GB
    peak, which does not fit next to the 9B LLM on 8 GB). Measured parity vs fp32 CPU: max |dp| 0.0026."""
    import laya
    import torch

    if device != "cuda" or not bf16_weights:
        return laya.load(repo, subfolder=subfolder, device=device)
    agent = laya.load(repo, subfolder=subfolder, device="cpu")
    agent.model.to(torch.bfloat16)
    agent.device, agent.dtype, agent.amp_enabled = torch.device("cuda"), torch.bfloat16, True
    agent.model.to(agent.device).eval()
    return agent


def temp_bucket(n: int) -> str:
    return "2" if n == 2 else "3-5" if n <= 5 else "6-10" if n <= 10 else "11+"


class LayaRouter(Router):
    name = "laya"

    def __init__(
        self,
        agent: Any,
        graph: ProceduralGraph,
        guard_mode: str = "choice",
        guard_cfg: dict | None = None,
        temperatures: dict[str, float] | None = None,
        use_none_option: bool = True,
        checkpoint: str = "",
        post_temperatures: dict[int, float] | None = None,
    ):
        self.agent = agent
        self.g = graph
        self.guard_mode = guard_mode  # "choice" (edge conditions) | "preset" (laya.guard_questions())
        self.guard_cfg = {**GUARD_DEFAULTS, **(guard_cfg or {})}
        self.use_none_option = use_none_option
        self.checkpoint = checkpoint
        cfg = agent.cfg
        # Reserve a few tokens for [SEP] etc. Budget is for the serialised state only.
        self.state_budget = int(cfg["max_len"]) - int(cfg["head_max_len"]) - 4
        if temperatures:
            self.set_temperatures(temperatures)
        # Per exact option count temperatures fitted on dev (Phase 5). Laya's own buckets ("3-5") are too coarse
        # for our nodes (3, 4 and 5 options), so Laya's choice temperatures are neutralised and we apply ours.
        self.post_temperatures = {int(k): float(v) for k, v in (post_temperatures or {}).items()}
        if self.post_temperatures:
            for b in ("2", "3-5", "6-10", "11+"):
                self.agent.temperature_by_options[f"choice:{b}"] = 1.0

    def warmup(self, n: int = 3) -> None:
        """First CUDA calls pay kernel/autotune cost (~1.7-2.6 s measured); keep it out of latency stats."""
        q = {"w": {"type": "choice", "instructions": "warm-up", "criteria": {"A": "one", "B": "two"}}}
        for _ in range(n):
            self.agent.predict({"ticket": "warm-up call", "facts": "none"}, q)

    @classmethod
    def from_config(
        cls, cfg: dict, graph: ProceduralGraph, device: str | None = None, finetuned: bool = False
    ) -> LayaRouter:
        c = cfg["laya"]
        repo, sub = c["repo"], (c.get("subfolder") or None)
        if finetuned:
            repo, sub = c["finetuned"], None
        agent = load_agent(repo, sub, device or c["device"], c.get("bf16_weights", True))
        router = cls(
            agent,
            graph,
            guard_mode=c.get("guard_mode", "choice"),
            guard_cfg=c.get("guard"),
            temperatures=c.get("temperatures"),
            use_none_option=c.get("none_option", True),
            checkpoint=f"{repo}/{sub or ''}",
            post_temperatures=c.get("post_temperatures"),
        )
        router.warmup()
        return router

    # -------------------------------------------------------------- helpers
    def set_temperatures(self, temps: dict[str, float]) -> None:
        """temps like {"choice:2": 1.3, "choice:3-5": 1.9}; overrides the shipped bucket values."""
        for k, v in temps.items():
            self.agent.temperature_by_options[k] = float(v)

    def count_tokens(self, state: dict) -> int:
        from laya.common import serialize_state

        tok = self.agent.tok
        text = serialize_state(state).replace(tok.mask_token, " ")
        return len(tok(text, add_special_tokens=False)["input_ids"])

    def render(self, state: AgentState) -> dict[str, str]:
        return render_state_for_laya(state, self.state_budget, self.count_tokens, extra=state.routing_extra)

    # -------------------------------------------------------------- routing
    def route(self, state: AgentState, node_id: str, edges: list[Edge], question: str) -> RoutingDecision:
        node = self.g.node(node_id)
        if (node.laya_questions or {}).get("kind") == "guard" and self.guard_mode == "preset":
            return self._guard_preset(state, edges)
        keys = option_keys(len(edges))
        criteria = {k: e.condition for k, e in zip(keys, edges, strict=True)}
        if self.use_none_option:
            criteria[NONE_KEY] = NONE_TEXT
        q = {"route": {"type": "choice", "instructions": question, "criteria": criteria}}
        rendered = self.render(state)
        t0 = time.perf_counter()
        res = self.agent.predict(rendered, q)
        latency = (time.perf_counter() - t0) * 1000
        a = res["answers"]["route"]
        if self.post_temperatures:
            a = self._recalibrate(a)
        key_to_target = {k: e.target for k, e in zip(keys, edges, strict=True)}
        choice = a["choice"]
        probs = {key_to_target.get(k, "NONE"): float(p) for k, p in a["probabilities"].items()}
        if choice == NONE_KEY:  # best real edge, but flagged as low confidence via raw_choice
            best = max(keys, key=lambda k: a["probabilities"].get(k, 0.0))
            target = key_to_target[best]
        else:
            target = key_to_target[choice]
        return RoutingDecision(
            target=target,
            confidence=float(a["confidence"]),
            answer_confidence=float(a["answer_confidence"]),
            probs=probs,
            router="laya",
            latency_ms=latency,
            raw_choice=choice,
            detail={
                "input_tokens": res.get("usage", {}).get("input_tokens"),
                "rendered_chars": len(json.dumps(rendered)),
                "truncated": "[...]" in rendered.get("ticket", ""),
                "n_options": len(criteria),
            },
        )

    def _recalibrate(self, a: dict) -> dict:
        keys = list(a["probabilities"])
        p = np.clip(np.array([a["probabilities"][k] for k in keys], dtype=float), 1e-6, None)
        T = self.post_temperatures.get(len(keys), 1.0)
        z = np.log(p / p.sum()) / T
        q = np.exp(z - z.max())
        q /= q.sum()
        ent = -float((q * np.log(q)).sum()) / math.log(len(q))
        i = int(q.argmax())
        return {
            **a,
            "choice": keys[i],
            "probabilities": dict(zip(keys, q.tolist(), strict=True)),
            "answer_confidence": float(q[i]),
            "confidence": 1.0 - ent,
            "raw_probabilities": a["probabilities"],
        }

    def _guard_preset(self, state: AgentState, edges: list[Edge]) -> RoutingDecision:
        import laya

        qs = {
            k: v for k, v in laya.guard_questions().items() if k in ("jailbreak", "prompt_injection", "harm_severity")
        }
        t0 = time.perf_counter()
        res = self.agent.predict({"prompt": state.ticket_text}, qs)
        latency = (time.perf_counter() - t0) * 1000
        a = res["answers"]
        p_attack = max(a["jailbreak"]["noul"], a["prompt_injection"]["noul"])
        harm = a["harm_severity"]["score"]
        flagged = p_attack >= self.guard_cfg["noul_threshold"] or harm >= self.guard_cfg["harm_threshold"]
        flag_edge = next(e for e in edges if e.target != "classify")
        clean_edge = next(e for e in edges if e.target == "classify")
        target = flag_edge.target if flagged else clean_edge.target
        conf = max(p_attack, 1 - p_attack)
        return RoutingDecision(
            target=target,
            confidence=conf,
            answer_confidence=conf,
            probs={flag_edge.target: p_attack, clean_edge.target: 1 - p_attack},
            router="laya",
            latency_ms=latency,
            raw_choice="flag" if flagged else "clean",
            detail={"p_attack": p_attack, "harm_severity": harm, "mode": "preset"},
        )
