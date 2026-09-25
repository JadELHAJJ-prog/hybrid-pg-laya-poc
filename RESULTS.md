# Results log

Per-phase log for the Hybrid PG PoC. Numbers here are measured; see each phase for the command that produced them.

## Phase 0 – Scaffold (DONE 2026-09-25)
**Built:** uv project (`pyproject.toml`, Python 3.11.x, `uv.lock`), `config.yaml`, `hpg` Typer CLI skeleton, pytest.
Claude Code tooling: `CLAUDE.md`; `.claude/agents/` (trace-analyst, phase-auditor, laya-api-scout);
`.claude/skills/` (phase-closeout, run-experiment, edit-procedural-graph); hooks in `.claude/settings.json`
(PreToolUse paid-API guard, PostToolUse ruff format/fix, SessionStart status); graphify CLAUDE.md section +
post-commit/post-checkout hooks.
**Key numbers:** `uv run pytest -q` → smoke test passes; `uv run hpg --help` lists commands.
**Deviations from plan:** repo root is `jev-x-pg/` (not a `hybrid-pg-poc/` subfolder), per user choice.
pytest disables ROS pytest plugins that leak in via the host `PYTHONPATH` (`/opt/ros/jazzy`).
**Open issues:** none.

## Phase 1 – Environment and VRAM probe (DONE 2026-09-25)
**Built:** `scripts/vram_probe.py` (pynvml sampler, loads LLM then Laya, times both, checks `ollama ps`),
`src/hpg/llm.py` (Ollama wrapper recording tokens, wall time and compute time), `ollama/Modelfile.qwen3.5-9b-text`.
**Versions (verified):** Ollama 0.32.1; `ollama` py 0.6.2; laya 0.3.20; torch 2.14.0+cu130; Python 3.11.
LLM = `qwen3.5-9b-text`, a local Ollama model built from `hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_M`
(5.68 GB text-only GGUF, mmproj vision projector left out) with `RENDERER/PARSER qwen3.5` copied from the official
`qwen3.5:9b` config. The official `qwen3.5:9b` tag is a single 6.59 GB GGUF with vision built in.
Laya = `convaiinnovations/laya`, `subfolder="typed-decisions"` (842 MB); English root also downloaded for Phase 5.
**Laya API verified from installed source** (laya-api-scout agent, file:line citations in its report):
`laya.load(repo, subfolder, device)` → `Agent`; `predict(state, questions)` returns per choice question
`choice, probabilities, confidence, answer_confidence, action`; `confidence = 1 − H(p)/log k` (not calibrated per
`laya/common.py:289-299`), `answer_confidence = max p` (the quantity temperature scaling / ECE use, `common.py:273-286`);
state dicts are `json.dumps`-serialised and truncated from the right beyond `max_len − head_max_len`
(typed-decisions 1024−256, English 512−192); temperatures live in `agent.temperature_by_options["choice:<bucket>"]`.
**Key numbers** (`uv run python scripts/vram_probe.py`, RTX 4060 Laptop 8188 MiB, desktop baseline 395 MiB):
| | value |
|---|---|
| after LLM load | 5785 MiB used, `ollama ps` 100% GPU |
| after LLM + Laya (bf16 weights) | 6826 MiB, peak 6826 MiB, headroom 1362 MiB, LLM still 100% GPU |
| Laya 5-option choice, GPU | p50 25.9 ms, p95 36.3 ms, torch max allocated 816 MiB |
| Laya same question, CPU fp32 (laya-api-scout run) | median 925 ms |
| LLM tiny call wall / compute | p50 1549 ms / 122 ms |
| LLM generation | 25.2 tok/s |
**Deviations from plan:**
1. *Stock Laya did not fit next to the LLM.* `laya.load(..., device="cuda")` keeps fp32 weights (bf16 autocast only);
   peak 2111 MiB → CUDA OOM at load → Laya silently fell back to CPU (1479 ms/call; VRAM peak 8170/8188 MiB).
   Fix: `hpg.routers.laya_router.load_agent` loads on CPU, casts weights to bf16, moves to CUDA. Parity vs fp32 CPU on 5
   states × 5 options: max |Δp| 0.0026, argmax agreement 5/5. No §2 fallback was needed.
2. The LLM is a locally built text-only model rather than a registry tag (the plan's §2 rule 1).
3. Gate field: the plan says gate on `confidence`; the library's own docs say `answer_confidence` is the calibrated
   one. Both are recorded; Phase 5 picks one on dev (`routing.gate`).
**Open issues:** Ollama 0.32.1 adds ~1.2–1.5 s of per-request `load_duration` for this model even while resident
(~0.6 s for qwen3:4b), so LLM wall time ≫ compute time. Both are reported separately in later phases.
Zero-shot Laya is weakly confident (answer_confidence ≈ 0.41 on an easy billing ticket), as the README warns.

