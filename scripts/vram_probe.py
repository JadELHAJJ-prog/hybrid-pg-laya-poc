"""Phase 1 probe: load the LLM (Ollama) and Laya together on the GPU, run calls, report peak VRAM + latency.

Usage: uv run python scripts/vram_probe.py [--laya-device cuda|cpu] [--subfolder typed-decisions|""]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import pynvml
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpg.llm import LLM  # noqa: E402


class VramSampler(threading.Thread):
    def __init__(self, period_s: float = 0.02):
        super().__init__(daemon=True)
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.total = pynvml.nvmlDeviceGetMemoryInfo(self.h).total / 2**20
        self.peak = 0.0
        self.period = period_s
        self._stop = threading.Event()

    def used(self) -> float:
        return pynvml.nvmlDeviceGetMemoryInfo(self.h).used / 2**20

    def run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self.used())
            time.sleep(self.period)

    def stop(self) -> None:
        self._stop.set()


def ollama_processor(model: str) -> str:
    out = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith(model):
            return " ".join(line.split()[3:6])
    return "not loaded"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--laya-device", default=None)
    ap.add_argument("--subfolder", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    dev = args.laya_device or cfg["laya"]["device"]
    sub = cfg["laya"]["subfolder"] if args.subfolder is None else args.subfolder
    s = VramSampler()
    s.start()
    base = s.used()
    report: dict = {"gpu_total_mib": round(s.total), "baseline_used_mib": round(base)}

    llm = LLM.from_config(cfg)
    msg = [{"role": "user", "content": "Reply with the single word: ready"}]
    t = time.perf_counter()
    llm.chat(msg, max_tokens=8)
    report["llm_cold_load_s"] = round(time.perf_counter() - t, 1)
    report["after_llm_used_mib"] = round(s.used())
    report["ollama_processor_llm_only"] = ollama_processor(cfg["llm"]["model"])

    import torch

    from hpg.routers.laya_router import load_agent

    t = time.perf_counter()
    agent = load_agent(cfg["laya"]["repo"], sub or None, dev, cfg["laya"].get("bf16_weights", True))
    report["laya_param_device_dtype"] = str(next(agent.model.parameters()).device) + " " + str(
        next(agent.model.parameters()).dtype
    )
    report["laya_load_s"] = round(time.perf_counter() - t, 1)
    q = {
        "route": {
            "type": "choice",
            "instructions": "What kind of request is this ticket?",
            "criteria": {
                "A": "The customer wants money back: a refund or a billing problem.",
                "B": "The customer asks where their order is.",
                "C": "The customer reports a technical problem with the app or website.",
                "D": "The request is none of the above.",
                "Z": "None of the listed conditions clearly applies.",
            },
        }
    }
    state = {"ticket": "Hi, I got charged twice for ORD-10031, please fix this.", "facts": "order_found: true"}
    agent.predict(state, q)  # warm-up
    lat = []
    for _ in range(20):
        t = time.perf_counter()
        r = agent.predict(state, q)
        lat.append((time.perf_counter() - t) * 1000)
    report["laya_device"] = dev
    report["laya_checkpoint"] = f"{cfg['laya']['repo']}/{sub}"
    report["laya_choice_answer"] = r["answers"]["route"]
    report["laya_ms_p50"] = round(statistics.median(lat), 1)
    report["laya_ms_p95"] = round(sorted(lat)[int(0.95 * len(lat)) - 1], 1)
    if dev == "cuda":
        report["laya_torch_max_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
    report["after_both_used_mib"] = round(s.used())

    llm_lat, llm_compute = [], []
    for _ in range(5):
        c = llm.chat(msg, max_tokens=8)
        llm_lat.append(c.wall_ms)
        llm_compute.append(c.compute_ms)
    report["llm_ms_p50_with_laya_loaded"] = round(statistics.median(llm_lat), 1)
    report["llm_compute_ms_p50"] = round(statistics.median(llm_compute), 1)
    report["ollama_processor_with_laya"] = ollama_processor(cfg["llm"]["model"])
    # a longer generation to stress KV cache while Laya is resident
    c = llm.chat([{"role": "user", "content": "Write 150 words about refunds."}], max_tokens=250)
    report["llm_250tok_wall_ms"] = round(c.wall_ms)
    report["llm_tokens_per_s"] = round(c.completion_tokens / max(c.compute_ms, 1) * 1000, 1)
    agent.predict(state, q)
    time.sleep(0.3)
    s.stop()
    report["peak_used_mib"] = round(s.peak)
    report["headroom_mib"] = round(s.total - s.peak)
    ok = "100% GPU" in report["ollama_processor_with_laya"] and str(agent.device).startswith(dev)
    report["laya_final_device"] = str(agent.device)
    report["verdict"] = "OK: both fit, LLM fully on GPU" if ok else "OVERFLOW: LLM offloaded to CPU"
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
