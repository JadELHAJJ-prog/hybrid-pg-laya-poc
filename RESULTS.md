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

## Phase 2 – World, tools, graph (DONE 2026-09-25)
**Built:** `scripts/make_world.py` → `data/orders.json` (60 orders in 8 policy buckets), `data/kb.json` (15 articles),
`data/service_status.json` (payments + login degraded); `src/hpg/tools.py` (7 deterministic tools, per-ticket ledger +
call log, BM25 over the KB); `src/hpg/schema.py`; `graphs/refunddesk_v1.yaml` (15 nodes, 24 edges, every edge has
condition/guidance/pitfalls); `src/hpg/graph_loader.py` (checks endpoints, no dead ends, reachability, every node
reaches END, cycles only through declared bounded back-edges).
**Key numbers:** `uv run pytest -q tests/test_tools.py tests/test_graph.py` → 13 passed; graph validation passes.
**Deviations from plan:**
1. Node executor `route` added for nodes that do no work (`classify`, `deny_with_alternative`).
2. Edge fields `max_traversals` / `on_exhausted` encode "max 1 retry then escalate_human" on
   final_review → draft_reply as data instead of engine code.
3. Node field `question` holds the routing question asked at multi-way nodes.
4. `lookup_order_delivery → draft_reply` also covers status `delivered` (plan text says "shipped or processing"), so
   the three delivery conditions cover every status.
5. `policy_check` conditions are written to be mutually exclusive, with duplicate charge taking priority.
**Open issues:** none.

## Phase 3 – Dataset (DONE 2026-09-25)
**Built:** `scripts/make_tickets.py` (hand-written templates + hand-written hard/adversarial cases, seeded, no LLM) →
`data/tickets.jsonl`; `src/hpg/gold.py` gold oracle (runs the real tools, so policy branches match `orders.json` by
construction); `scripts/validate_dataset.py`.
**Key numbers** (`uv run python scripts/validate_dataset.py` → OK): 220 tickets, dev 70 / test 150 (stratified by gold
path); every non-exempt edge ≥ 11 gold examples; hard 52 (24%: sarcasm, two issues, vague, missing id, wrong id, long
1.3k-word rambles, fact-driven); adversarial 16 (7%: 12 injections + 4 threats/abuse). Gold final actions:
escalate_human 45, deny 34, refund_standard 30, technical_reply 30, refund_duplicate 27, delivery_update 27,
ask_for_info 27.
**Deviations from plan:** `final_review → draft_reply` is exempt from the ≥8 rule because whether it is taken depends
on generated text, so no static gold label exists. The hard (24%) and adversarial (7%) shares are above the plan's
~15% / ~5%. `gold_category` includes `adversarial`.
**Open issues:** for two-issue tickets, gold follows the issue the customer explicitly asks to have resolved;
these are tagged `two_issues` so they can be reported separately.

