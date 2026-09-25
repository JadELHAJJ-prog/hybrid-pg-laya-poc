"""Assemble the PoC results (test experiments, dev calibration, failure analysis) for the dashboard."""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from hpg.eval.calibrate import ece
from hpg.eval.metrics import final_action_from_trace, load_tickets
from hpg.tracing import read_trace

EXPS = {
    "E1": "LLM routes everything",
    "E2": "Laya only (zero-shot)",
    "E3": "Hybrid, zero-shot Laya",
    "E4": "Hybrid, Laya on CPU",
    "E5": "Hybrid, fine-tuned Laya",
}
DECISION_NODES = ("guard", "classify", "lookup_order", "policy_check", "lookup_order_delivery", "final_review")


def _stamp(root: Path) -> float:
    files = list((root / "reports").glob("*.json")) + list((root / "runs").glob("E*_test"))
    return max((f.stat().st_mtime for f in files), default=0.0)


def build_results(root: Path, cfg: dict) -> dict:
    return _build(str(root), json.dumps(cfg, default=str), _stamp(root))


@lru_cache(maxsize=4)
def _build(root_s: str, cfg_s: str, _stamp_v: float) -> dict:
    root, cfg = Path(root_s), json.loads(cfg_s)
    reports = root / "reports"
    summary = (
        json.loads((reports / "summary_test.json").read_text()) if (reports / "summary_test.json").exists() else {}
    )
    tickets = load_tickets(root / cfg["paths"]["tickets"])
    out: dict = {"experiments": {}, "labels": EXPS, "n_test": sum(t["split"] == "test" for t in tickets.values())}
    for e, s in summary.items():
        out["experiments"][e] = {k: v for k, v in s.items() if not isinstance(v, dict)} | {
            "routing_acc_by_node": s.get("routing_acc_by_node", {}),
            "route_latency_by_router": s.get("route_latency_by_router", {}),
        }
    for tag, name in (("", "dev_zero_shot"), ("_finetuned", "dev_finetuned")):
        p = reports / f"calibration_dev{tag}.json"
        if p.exists():
            out[name] = json.loads(p.read_text())
    out["test_calibration"], out["failures"] = {}, {}
    for e in summary:
        run = root / cfg["paths"]["runs"] / f"{e}_test"
        acc, conf, fails, fb = [], [], Counter(), 0
        examples: dict[str, list[str]] = {}
        for f in sorted(run.glob("T*.jsonl")):
            steps = read_trace(f)
            t = tickets[f.stem]
            gold = t["gold_path"]
            first = None
            for s in steps:
                if s.get("type") != "route" or s.get("router") == "deterministic":
                    continue
                k = len(s["path"])
                on = gold[:k] == s["path"] and k < len(gold)
                if on and s["node"] in DECISION_NODES:
                    if s["router"] == "laya":
                        acc.append(s["target"] == gold[k])
                        conf.append(s.get("answer_confidence") or s["confidence"])
                    elif s["router"] == "llm_fallback":
                        fb += 1
                if on and first is None and s["target"] != gold[k]:
                    first = s
            if final_action_from_trace(steps) != t["gold_final_action"]:
                key = f"{first['node']} ({'LLM' if first['router'] != 'laya' else 'Laya'})" if first else "other"
                fails[key] += 1
                examples.setdefault(key, []).append(f.stem)
        a, c = np.array(acc, float), np.array(conf, float)
        out["test_calibration"][e] = (
            {
                "n": len(a),
                "acc": float(a.mean()),
                "mean_conf": float(c.mean()),
                "ece": ece(c, a.astype(bool)),
                "fallbacks": fb,
            }  # fmt: skip
            if len(a)
            else {"n": 0, "fallbacks": fb}
        )
        out["failures"][e] = {"total": sum(fails.values()), "by_first_divergence": dict(fails.most_common()),
                              "examples": {k: v[:4] for k, v in examples.items()}}  # fmt: skip
    out["config"] = {
        "llm": cfg["llm"]["model"],
        "laya_zero_shot": f"{cfg['laya']['repo']}/{cfg['laya'].get('subfolder') or '(English root)'}",
        "tau_e3": cfg["routing"]["tau"],
        "tau_e5": cfg.get("finetuned_calibration", {}).get("tau"),
        "gate": cfg["routing"].get("gate"),
    }
    return out
