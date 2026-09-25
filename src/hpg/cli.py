"""hpg command-line interface."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import typer
import yaml

app = typer.Typer(help="Hybrid System-1/System-2 Procedural Graph PoC (RefundDesk).", no_args_is_help=True)
ROOT = Path.cwd()

# Experiment matrix (plan §9). E5 uses a fine-tuned checkpoint path given in config laya.finetuned.
EXPERIMENTS = {
    "E1": {"router": "llm", "device": None},
    "E2": {"router": "laya", "device": None},
    "E3": {"router": "hybrid", "device": None},
    "E4": {"router": "hybrid", "device": "cpu"},
    "E5": {"router": "hybrid", "device": None, "finetuned": True},
}


def load_cfg(path: str = "config.yaml") -> dict:
    return yaml.safe_load((ROOT / path).read_text())


def load_tickets(cfg: dict) -> list[dict]:
    p = ROOT / cfg["paths"]["tickets"]
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def build(cfg: dict, router_name: str, device: str | None = None, finetuned: bool = False):
    from hpg.engine import Engine
    from hpg.graph_loader import load_graph
    from hpg.llm import LLM
    from hpg.routers.hybrid_router import HybridRouter
    from hpg.routers.llm_router import LLMRouter
    from hpg.tools import World

    graph = load_graph(ROOT / cfg["paths"]["graph"])
    world = World.from_config(cfg, ROOT)
    llm = LLM.from_config(cfg)
    llm_router = LLMRouter(llm, think=cfg["llm"].get("think_routing", False))
    router = llm_router
    if router_name in ("laya", "hybrid"):
        from hpg.routers.laya_router import LayaRouter

        router = LayaRouter.from_config(cfg, graph, device=device, finetuned=finetuned)
        if router_name == "hybrid":
            r = cfg["routing"]
            router = HybridRouter(router, llm_router, tau=r["tau"], gate=r.get("gate", "confidence"))
    return Engine(graph, world, router, llm, cfg["routing"]["max_steps"], cfg["llm"]["max_tool_rounds"])


@app.callback()
def main() -> None:
    """Hybrid System-1/System-2 Procedural Graph PoC (RefundDesk)."""


@app.command()
def version() -> None:
    """Print package version."""
    from importlib.metadata import version as v

    typer.echo(v("hpg"))


@app.command()
def run(
    ticket: str = typer.Option(..., help="ticket id, e.g. T0001"),
    router: str = typer.Option("hybrid", help="llm | laya | hybrid"),
    device: str | None = typer.Option(None, help="override laya.device (cuda|cpu)"),
    show_reply: bool = typer.Option(True),
) -> None:
    """Run one ticket and print the path with per-step router and confidence."""
    from hpg.tracing import Tracer

    cfg = load_cfg()
    t = next((x for x in load_tickets(cfg) if x["ticket_id"] == ticket), None)
    if t is None:
        raise typer.BadParameter(f"unknown ticket {ticket}")
    eng = build(cfg, router, device)
    out = ROOT / cfg["paths"]["runs"] / f"single_{router}" / f"{ticket}.jsonl"
    tracer = Tracer(out)
    st = eng.run(t, tracer)
    typer.echo(f"{ticket} [{t['split']}] gold={t['gold_final_action']}\n  {t['text'][:160]}")
    for r in tracer.records:
        if r["type"] == "route":
            ok = ""
            gold = t["gold_path"]
            k = len(r["path"])
            if gold[:k] == r["path"] and k < len(gold):
                ok = "  OK" if gold[k] == r["target"] else f"  WRONG (gold {gold[k]})"
            extra = ""
            if r["router"] == "llm_fallback":
                extra = f" (laya said {r['laya_target']} @ {r['laya_confidence']:.2f})"
            typer.echo(
                f"  {r['node']:>22} -> {r['target']:<22} router={r['router']:<12} "
                f"conf={r['confidence']:.2f} {r['latency_ms']:.0f}ms{extra}{ok}"
            )
        elif r["type"] in ("node", "llm_node") and r.get("executor") in ("llm", "tool"):
            typer.echo(f"  {r['node']:>22} [{r['executor']}] {r.get('ms', 0):.0f}ms")
    end = tracer.records[-1]
    typer.echo(f"  path: {' > '.join(st.path)}")
    typer.echo(
        f"  path matches gold: {st.path == t['gold_path']}   ledger: {st.ledger}   error: {end['error']}"
    )
    if show_reply and (st.draft_reply or st.info_request):
        typer.echo(f"  reply: {st.draft_reply or st.info_request}")
    typer.echo(f"  trace: {out}")


def run_experiment(cfg: dict, exp: str, split: str, limit: int | None = None, resume: bool = True) -> Path:
    from hpg.tracing import Tracer

    spec = EXPERIMENTS[exp]
    tickets = [t for t in load_tickets(cfg) if t["split"] == split][:limit]
    out_dir = ROOT / cfg["paths"]["runs"] / f"{exp}_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.snapshot.yaml").write_text(yaml.safe_dump({**cfg, "experiment": {exp: spec}}))
    import threading

    import pynvml

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    peak = {"mib": 0.0}
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            peak["mib"] = max(peak["mib"], pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**20)
            time.sleep(0.05)

    threading.Thread(target=sample, daemon=True).start()
    eng = build(cfg, spec["router"], spec["device"], spec.get("finetuned", False))
    t0 = time.time()
    done = 0
    for i, t in enumerate(tickets, 1):
        f = out_dir / f"{t['ticket_id']}.jsonl"
        if resume and f.exists() and '"type": "end"' in f.read_text():
            continue
        eng.run(t, Tracer(f))
        done += 1
        if i % 10 == 0 or i == len(tickets):
            typer.echo(f"  {exp}/{split}: {i}/{len(tickets)} ({time.time() - t0:.0f}s)")
    stop.set()
    vf = out_dir / "vram.json"
    prev = json.loads(vf.read_text())["peak_used_mib"] if vf.exists() else 0
    vf.write_text(json.dumps({"peak_used_mib": round(max(prev, peak["mib"])), "note": "whole GPU, nvml"}))
    return out_dir


@app.command("eval")
def eval_cmd(
    exp: list[str] = typer.Option(None, help="experiment ids (E1..E5); repeatable"),
    split: str = typer.Option("test"),
    all_: bool = typer.Option(False, "--all", help="run E1-E4 on test, then write the report"),
    limit: int | None = typer.Option(None),
    fresh: bool = typer.Option(False, help="delete existing traces first"),
) -> None:
    """Run experiments and print metrics. `--all` reproduces the RESULTS.md numbers from scratch."""
    from hpg.eval.metrics import load_tickets as lt
    from hpg.eval.metrics import summarize

    cfg = load_cfg()
    exps = ["E1", "E2", "E3", "E4"] if all_ else (exp or ["E3"])
    tick = lt(ROOT / cfg["paths"]["tickets"])
    summaries = {}
    for e in exps:
        d = ROOT / cfg["paths"]["runs"] / f"{e}_{split}"
        if fresh and d.exists():
            shutil.rmtree(d)
        d = run_experiment(cfg, e, split, limit)
        summaries[e] = summarize(d, tick)
        typer.echo(json.dumps({e: summaries[e]}, indent=1, default=float))
    out = ROOT / cfg["paths"]["runs"] / f"summary_{split}.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    out.write_text(json.dumps({**prev, **summaries}, indent=1, default=float))
    if all_:
        from hpg.eval.report import write_report

        write_report(cfg, split)


@app.command()
def calibrate(
    stage: str = typer.Option("all", help="collect | score | analyze | all"),
    split: str = typer.Option("dev"),
    limit: int | None = typer.Option(None),
) -> None:
    """Phase 5 (dev only): collect decisions, replay through Laya checkpoints, fit temperatures, choose tau."""
    if split == "test":
        raise typer.BadParameter("calibration/tuning on the test split is not allowed (plan §12.3)")
    from hpg.eval import calibrate as C

    cfg = load_cfg()
    if stage in ("collect", "all"):
        typer.echo(f"decisions -> {C.collect(cfg, ROOT, split, limit)}")
    if stage in ("score", "all"):
        for name, sub in cfg["laya"].get("candidates", {"typed": "typed-decisions", "english": ""}).items():
            typer.echo(f"laya {name} -> {C.score_laya(cfg, ROOT, name, sub or None, split)}")
    if stage in ("analyze", "all"):
        from hpg.eval.report import calibration_report

        calibration_report(cfg, ROOT, split)


if __name__ == "__main__":
    app()
