"""Reports: Phase 5 calibration (dev) and Phase 6 experiment tables/plots (test)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hpg.eval import calibrate as C  # noqa: E402

REPORTS = "reports"


def _load(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _stats(ev: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "acc": float(ev["correct"].mean()),
        "ece_answer_conf": C.ece(ev["answer_conf"], ev["correct"]),
        "mean_answer_conf": float(ev["answer_conf"].mean()),
        "none_won_rate": float(ev["none_won"].mean()),
    }


def calibration_report(cfg: dict, root: Path, split: str = "dev", write_config: bool = True) -> dict:
    calib = root / cfg["paths"]["runs"] / f"calib_{split}"
    out_dir = root / REPORTS
    out_dir.mkdir(exist_ok=True)
    cands = cfg["laya"].get("candidates", {"typed": "typed-decisions", "english": ""})
    summary: dict = {"split": split, "candidates": {}}
    taus = np.round(np.linspace(0.0, 1.0, 101), 2)
    fig_rel, axes_rel = plt.subplots(1, len(cands), figsize=(5 * len(cands), 4.2), squeeze=False)
    fig_tau, axes_tau = plt.subplots(1, len(cands), figsize=(5.5 * len(cands), 4.2), squeeze=False)
    for ci, (name, sub) in enumerate(cands.items()):
        recs = [r for r in _load(calib / f"laya_{name}.jsonl") if r["node"] in C.ROUTING_NODES]
        llm_acc = float(np.mean([r["llm_correct"] for r in recs]))
        raw = C.evaluate_records(recs, None)
        shipped_T = {len(C.option_probs(r)[1]): r["shipped_T"] or 1.0 for r in recs}
        shipped = C.evaluate_records(recs, shipped_T)
        cv, temps_all = C.cv_temps(recs)
        s = {
            "subfolder": sub,
            "n_decisions": len(recs),
            "llm_acc_teacher_forced": llm_acc,
            "raw_T1": _stats(raw),
            "shipped_T": {**_stats(shipped), "T": shipped_T},
            "fitted_T_cv5": {**_stats(cv), "T_all_dev": temps_all},
            "acc_by_node": {},
            "gates": {},
        }
        nodes = np.array([r["node"] for r in recs])
        for n in C.ROUTING_NODES:
            m = nodes == n
            if m.any():
                s["acc_by_node"][n] = {"laya": float(cv["correct"][m].mean()),
                                       "llm": float(cv["llm_correct"][m].mean()), "n": int(m.sum())}  # fmt: skip
        ax = axes_tau[0][ci]
        for gate, style in (("answer_confidence", "-"), ("confidence", "--")):
            curve = C.hybrid_curve(cv, gate, taus)
            pick = C.choose_tau(curve, llm_acc)
            s["gates"][gate] = {"chosen": pick, "curve": curve}
            ax.plot([c["tau"] for c in curve], [c["hybrid_acc"] for c in curve], style, label=f"hybrid acc ({gate})")
            ax.plot([c["tau"] for c in curve], [c["coverage"] for c in curve], style, alpha=0.5,
                    label=f"Laya coverage ({gate})")  # fmt: skip
        ax.axhline(llm_acc, color="k", lw=0.8, label="LLM-only acc (teacher-forced)")
        ax.axhline(llm_acc - 0.02, color="k", lw=0.8, ls=":", label="LLM − 2 pts")
        ax.set(title=f"{name}: accuracy / coverage vs tau (dev, CV-calibrated)", xlabel="tau", ylim=(0, 1.02))
        ax.legend(fontsize=7)
        ax = axes_rel[0][ci]
        for lab, ev in (("raw T=1", raw), ("shipped T", shipped), ("fitted T (5-fold CV)", cv)):
            pts = C.reliability(ev["answer_conf"], ev["correct"])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=3,
                    label=f"{lab} ECE={C.ece(ev['answer_conf'], ev['correct']):.3f}")  # fmt: skip
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set(title=f"{name}: reliability (answer_confidence)", xlabel="confidence", ylabel="accuracy")
        ax.legend(fontsize=7)
        summary["candidates"][name] = s
    fig_rel.tight_layout()
    fig_rel.savefig(out_dir / f"reliability_{split}.png", dpi=120)
    fig_tau.tight_layout()
    fig_tau.savefig(out_dir / f"tau_sweep_{split}.png", dpi=120)
    plt.close("all")

    # guard preset vs choice on dev
    recs = _load(calib / f"laya_{next(iter(cands))}.jsonl")
    g = [r for r in recs if r["node"] == "guard" and "guard_preset" in r]
    if g:
        flag_gold = np.array([r["gold_target"] == "escalate_human" for r in g])

        def rates(pred):
            pred = np.array(pred)
            return {"recall_adv": float(pred[flag_gold].mean()) if flag_gold.any() else None,
                    "fpr_clean": float(pred[~flag_gold].mean())}  # fmt: skip

        summary["guard_dev"] = {
            "preset": rates([r["guard_preset"]["target"] == "escalate_human" for r in g]),
            "choice_laya": rates([r["laya"]["target"] == "escalate_human" for r in g]),
            "llm": rates([r["llm_target"] == "escalate_human" for r in g]),
        }

    # decision: checkpoint with best CV accuracy; gate with highest coverage at the chosen tau
    best = max(summary["candidates"], key=lambda n: summary["candidates"][n]["fitted_T_cv5"]["acc"])
    b = summary["candidates"][best]
    gate = max(b["gates"], key=lambda gname: b["gates"][gname]["chosen"]["coverage"])
    choice = b["gates"][gate]["chosen"]
    summary["decision"] = {
        "checkpoint": best,
        "subfolder": b["subfolder"],
        "gate": gate,
        "tau": choice["tau"],
        "dev_hybrid_acc_sim": choice["hybrid_acc"],
        "dev_fallback_rate_sim": choice["fallback_rate"],
        "dev_llm_acc": b["llm_acc_teacher_forced"],
        "post_temperatures": b["fitted_T_cv5"]["T_all_dev"],
        "within_2pts": choice["hybrid_acc"] >= b["llm_acc_teacher_forced"] - 0.02,
    }
    gd = summary.get("guard_dev")
    if gd:
        def j(m):  # Youden's J on dev: recall on adversarial minus false-positive rate on clean
            return (m["recall_adv"] or 0.0) - m["fpr_clean"]

        # The tau sweep above simulates the choice-mode guard, so only switch to the preset if it is clearly better
        # AND does not flag many more clean tickets (each false flag escalates a normal customer).
        better = j(gd["preset"]) > j(gd["choice_laya"])
        safe = gd["preset"]["fpr_clean"] <= gd["choice_laya"]["fpr_clean"] + 0.05
        summary["decision"]["guard_mode"] = "preset" if (better and safe) else "choice"
    slim = json.loads(json.dumps(summary, default=float))
    for c in slim["candidates"].values():
        for gname in c["gates"]:
            c["gates"][gname].pop("curve")
    (out_dir / f"calibration_{split}.json").write_text(json.dumps(slim, indent=1))
    print(json.dumps(slim, indent=1))
    if write_config and split == "dev":
        _write_config(root / "config.yaml", summary["decision"])
    return summary


def _write_config(path: Path, d: dict) -> None:
    s = path.read_text()
    sub = d["subfolder"] or '""'
    s = re.sub(r"(?m)^  subfolder: .*$", f"  subfolder: {sub}   # chosen on dev in Phase 5", s)
    s = re.sub(r"(?m)^  tau: .*$", f"  tau: {d['tau']}   # chosen on dev in Phase 5 (reports/calibration_dev.json)", s)
    if re.search(r"(?m)^  gate: ", s):
        s = re.sub(r"(?m)^  gate: .*$", f"  gate: {d['gate']}", s)
    else:
        s = re.sub(r"(?m)^(  tau: .*)$", rf"\1\n  gate: {d['gate']}   # confidence | answer_confidence", s)
    temps = ", ".join(f"{k}: {v:.3f}" for k, v in sorted(d["post_temperatures"].items()))
    line = f"  post_temperatures: {{{temps}}}   # per option count, fitted on dev (Phase 5)"
    if re.search(r"(?m)^  post_temperatures: ", s):
        s = re.sub(r"(?m)^  post_temperatures: .*$", line, s)
    else:
        s = re.sub(r"(?m)^(  bf16_weights: .*)$", rf"\1\n{line}", s)
    if "guard_mode" in d:
        gm = f"  guard_mode: {d['guard_mode']}   # choice | preset, chosen on dev (Phase 5)"
        if re.search(r"(?m)^  guard_mode: ", s):
            s = re.sub(r"(?m)^  guard_mode: .*$", gm, s)
        else:
            s = re.sub(r"(?m)^(  bf16_weights: .*)$", rf"\1\n{gm}", s)
    path.write_text(s)


# ----------------------------------------------------------------------------- Phase 6
def write_report(cfg: dict, split: str = "test", root: Path | None = None) -> Path:
    """Tables + plots for E1-E4(E5) from traces; writes reports/results_<split>.md (embedded into RESULTS.md)."""
    from hpg.eval.metrics import decisions_frame, load_tickets, summarize, ticket_frame

    root = root or Path.cwd()
    runs = root / cfg["paths"]["runs"]
    out_dir = root / REPORTS
    out_dir.mkdir(exist_ok=True)
    tick = load_tickets(root / cfg["paths"]["tickets"])
    exps = [e for e in ("E1", "E2", "E3", "E4", "E5") if (runs / f"{e}_{split}").exists()]
    S = {e: summarize(runs / f"{e}_{split}", tick) for e in exps}
    lines = [f"### Experiment results ({split} split, n={S[exps[0]]['n_tickets']} tickets)\n"]
    hdr = ("| metric | " + " | ".join(exps) + " |", "|---|" + "---|" * len(exps))
    lines += hdr
    rows = [
        ("routing accuracy (4 routing nodes)", "routing_acc", "{:.3f}"),
        ("  scored routing decisions (n)", "routing_n", "{}"),
        ("decision accuracy incl. guard", "decision_acc_incl_guard", "{:.3f}"),
        ("full-path exact match", "path_exact", "{:.3f}"),
        ("final-action accuracy", "final_action_acc", "{:.3f}"),
        ("guard recall (adversarial)", "guard_recall_adv", "{:.3f}"),
        ("guard FPR (clean)", "guard_fpr_clean", "{:.3f}"),
        ("routing latency p50 ms", "route_p50_ms", "{:.0f}"),
        ("routing latency p95 ms", "route_p95_ms", "{:.0f}"),
        ("end-to-end p50 s", "e2e_p50_ms", "{:.1f}", 1e-3),
        ("end-to-end p95 s", "e2e_p95_ms", "{:.1f}", 1e-3),
        ("LLM calls / ticket", "llm_calls_per_ticket", "{:.2f}"),
        ("  of which routing", "llm_calls_routing_per_ticket", "{:.2f}"),
        ("LLM tokens / ticket", "llm_tokens_per_ticket", "{:.0f}"),
        ("  routing tokens", "llm_tokens_routing_per_ticket", "{:.0f}"),
        ("  in-node tokens", "llm_tokens_nodes_per_ticket", "{:.0f}"),
        ("fallback rate (of Laya decisions)", "fallback_rate", "{:.3f}"),
        ("final_review first-pass rate", "review_first_pass_rate", "{:.3f}"),
        ("engine errors", "errors", "{}"),
    ]
    for r in rows:
        label, key, fmt = r[:3]
        scale = r[3] if len(r) > 3 else 1
        vals = []
        for e in exps:
            v = S[e][key]
            vals.append("n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v * scale))
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines.append("\nRouting latency by router (ms, p50 / p95 / n):\n")
    for e in exps:
        br = S[e]["route_latency_by_router"]
        lines.append(
            f"- {e}: " + "; ".join(f"{k} {v['p50']:.0f} / {v['p95']:.0f} / {int(v['n'])}" for k, v in br.items())
        )
    lines.append("\nRouting accuracy by node:\n")
    nodes = sorted({n for e in exps for n in S[e]["routing_acc_by_node"]})
    lines += ["| node | " + " | ".join(exps) + " |", "|---|" + "---|" * len(exps)]
    for n in nodes:
        lines.append(
            f"| {n} | " + " | ".join(f"{S[e]['routing_acc_by_node'].get(n, float('nan')):.3f}" for e in exps) + " |"
        )
    # slices
    lines.append("\nFinal-action accuracy by slice:\n")
    lines += ["| slice | n | " + " | ".join(exps) + " |", "|---|---|" + "---|" * len(exps)]
    tfs = {e: ticket_frame(runs / f"{e}_{split}", tick) for e in exps}
    base = tfs[exps[0]]
    slices = {"all": np.ones(len(base), bool), "hard": base.tags.str.contains("hard").values,
              "adversarial": base.tags.str.contains("adversarial").values,
              "long": base.tags.str.contains("long").values,
              "two_issues": base.tags.str.contains("two_issues").values}  # fmt: skip
    for cat in sorted(base.category.unique()):
        slices[f"cat={cat}"] = (base.category == cat).values
    for sname, m in slices.items():
        ids = set(base.ticket_id[m])
        vals = [f"{tfs[e][tfs[e].ticket_id.isin(ids)].action_correct.mean():.3f}" for e in exps]
        lines.append(f"| {sname} | {len(ids)} | " + " | ".join(vals) + " |")
    # plots
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for e in exps:
        d = decisions_frame(runs / f"{e}_{split}", tick)
        d = d[d.node.isin(C.ROUTING_NODES)]
        ax[0].hist(np.log10(d.latency_ms.clip(lower=1)), bins=40, alpha=0.5, label=e)
    ax[0].set(title="routing latency per decision (log10 ms)", xlabel="log10(ms)")
    ax[0].legend()
    ax[1].bar(exps, [S[e]["routing_acc"] for e in exps], color="#4c72b0")
    ax2 = ax[1].twinx()
    ax2.plot(exps, [S[e]["llm_calls_per_ticket"] for e in exps], "o-", color="#dd8452")
    ax[1].set(title="routing accuracy (bars) vs LLM calls/ticket (line)", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(out_dir / f"experiments_{split}.png", dpi=120)
    plt.close(fig)
    lines.append(f"\n![experiments](reports/experiments_{split}.png)\n")
    md = "\n".join(lines) + "\n"
    out = out_dir / f"results_{split}.md"
    out.write_text(md)
    (out_dir / f"summary_{split}.json").write_text(json.dumps(S, indent=1, default=float))
    print(md)
    return out
