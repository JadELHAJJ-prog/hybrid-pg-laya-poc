---
name: run-experiment
description: Use when running any RefundDesk evaluation (E1-E5, dev calibration sweeps, single-ticket debug runs) or when regenerating RESULTS.md numbers. Covers the commands, VRAM hygiene, split discipline, and where outputs go.
---

# Running experiments

## Commands
- One ticket: `uv run hpg run --ticket T0001 --router {llm|laya|hybrid}` prints the path with per-step router + confidence.
- Experiment: `uv run hpg eval --exp E3 --split test`; everything: `uv run hpg eval --all`.
- Calibration / tau sweep (dev only): `uv run hpg calibrate --split dev`.
- Traces: `runs/<experiment>/<ticket_id>.jsonl`. Metrics are computed from traces only.

## Split discipline (hard rule)
- `dev` (70 tickets): used for tau, temperatures, prompt/edge-text iteration, and checkpoint choice.
- `test` (150 tickets): final numbers only. If you catch yourself changing config after looking at test results, stop and tell the user.

## Before a run
1. `nvidia-smi` should show the GPU mostly free. Unload stray models with `ollama stop <model>`.
2. Check that `config.yaml` has the intended llm tag, laya checkpoint, device, and tau. The run stores a copy of it in `runs/<exp>/config.snapshot.yaml`.
3. For latency numbers, run a warm-up call first and exclude it from the stats.

## After a run
- Use the `trace-analyst` agent for failure analysis.
- Record which §2 VRAM fallback (if any) was active.
