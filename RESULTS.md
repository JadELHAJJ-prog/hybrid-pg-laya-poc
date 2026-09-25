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

## Phase 4 – Engine and routers (DONE 2026-09-25)
**Built:** `engine.py` (graph walker: tool / llm / route / laya_check executors; deterministic routing for single-edge
nodes; max_steps 20; bounded back-edge final_review → draft_reply with max 1 retry, then escalate_human; every
step traced), `routers/{base,llm_router,laya_router,hybrid_router}.py`, `llm_nodes.py` (ask_for_info, draft_reply,
investigate_technical with native Ollama tool calling, max 4 rounds; guidance + pitfalls injected from the
incoming edge and all upstream decisions), `tracing.py`, `state_render.py`, `eval/metrics.py`, CLI `hpg run` / `hpg eval`.
- LLMRouter: opaque letters A.. plus `Z: none applies`, JSON-schema `enum` output, thinking off, confidence =
  logprob of the chosen letter renormalised over the allowed letters.
- LayaRouter: one `choice` question per decision node (opaque keys, edge conditions as option text, `Z` option);
  state = `{"ticket", "facts"[, "reply"]}`, ticket pre-truncated head+tail with Laya's own tokenizer so facts
  always survive; optional `guard_questions()` preset mode for the guard; warm-up calls at load.
- HybridRouter: Laya first; LLM when gate score < tau or Laya picks `Z`; both answers recorded in the trace.
**Key numbers:** `uv run pytest -q` → 25 passed (includes: gold-following router reproduces all 220 gold paths and
final actions through the real engine + metrics; retry → escalate rule; router logic with fake models).
`hpg run` on T0013 (refund_standard) is correct under llm, laya and hybrid. On T0035 (technical ticket that mentions an
order id), Laya zero-shot routes classify → lookup_order_delivery (wrong, answer_confidence low), and hybrid recovers via
fallback. Observed per-step routing wall time: LLM ≈ 2.0–2.6 s, Laya ≈ 30–100 ms (warm). Zero-shot Laya entropy
`confidence` is 0.03–0.16 on these steps, so at the placeholder tau 0.5 every hybrid step falls back.
**Deviations from plan:** there is no separate `checks.py`: guard / final_review are laya_check nodes routed by the
active router (the LayaRouter guard preset mode lives in `laya_router.py`). Under E1 the LLM also decides guard and
final_review, matching "LLMRouter everywhere". When the LLM answers `Z`, it takes the node's last edge with
confidence 0.
**Open issues:** the first Laya CUDA call costs 1.7–2.6 s, so a warm-up was added.

## Phase 5 – Calibration and threshold tuning, dev only (DONE 2026-09-25)
**Method** (`uv run hpg calibrate --split dev`; `--split test` is refused):
1. Collect: walk all 70 dev tickets along their gold path with the real LLM nodes. At every multi-way decision,
   record the LLMRouter answer plus a snapshot of the exact routing input: 264 decisions over guard, classify,
   lookup_order, policy_check, lookup_order_delivery and final_review.
2. Replay each snapshot through both Laya checkpoints with the choice temperatures neutralised (T=1) to get raw
   probabilities.
3. Fit one temperature per exact option count (3, 4 or 5 options including `Z`) by NLL. The ECE for the fitted
   temperatures comes from 5-fold CV grouped by ticket, so the fitting and evaluation data are disjoint. Simulate
   hybrid accuracy vs tau from these CV-calibrated probabilities (Laya if score ≥ tau and not `Z`, else the
   recorded LLM decision). Pick the smallest tau with hybrid accuracy ≥ LLM − 2 pts.
**Key numbers** (`reports/calibration_dev.json`, `reports/reliability_dev.png`, `reports/tau_sweep_dev.png`):
| dev, 264 decisions | typed-decisions | English root |
|---|---|---|
| Laya accuracy (routing semantics) | 0.814 | **0.867** |
| ECE answer_confidence: raw T=1 / shipped T / fitted T (CV) | 0.210 / 0.326 / 0.074 | 0.137 / 0.285 / **0.044** |
| fitted T per option count {3,4,5} | 0.65 / 0.40 / 0.38 | 0.57 / 0.74 / 0.78 |
| chosen tau, answer_confidence gate → sim. hybrid acc / Laya coverage | 0.89 → 0.970 / 34.8% | **0.87 → 0.970 / 56.4%** |
| chosen tau, entropy `confidence` gate → sim. hybrid acc / coverage | 0.68 → 0.970 / 33.0% | 0.67 → 0.970 / 51.9% |
| LLMRouter accuracy, same decisions | 0.989 | 0.989 |
Per-node accuracy, Laya-English vs LLM: guard 0.96/0.97, classify 0.88/0.98, lookup_order 0.97/1.00, policy_check
0.67/1.00, lookup_order_delivery 0.62/1.00, final_review 0.85/1.00.
Guard on the 70 dev guard decisions (5 adversarial): `guard_questions()` preset (jailbreak/prompt_injection noul
≥ 0.5 or harm_severity ≥ 1.5) recall 0.80 / FPR 0.34; Laya 2-option choice recall 0.20 / FPR 0.00; LLM recall
1.00 / FPR 0.03.
**Decision written to config.yaml:** checkpoint = English root (`subfolder: ""`), gate = `answer_confidence`,
tau = 0.87, post-hoc temperatures {3: 0.575, 4: 0.742, 5: 0.783}, guard_mode = choice (the preset flags a third of
clean dev tickets, and the tau sweep was simulated with the choice guard).
**Findings:** (a) Both checkpoints are *under*-confident on these decisions, so the fitted T < 1 sharpens them. The
shipped temperatures (T ≈ 1.76 for 3–5 options) make calibration worse. (b) Zero-shot, the English root beats the
fine-tuned typed-decisions checkpoint here, mostly on final_review (0.85 vs 0.54). (c) Laya is weakest on nodes that
need reading facts (policy_check, lookup_order_delivery). (d) Even at the chosen tau, ~44% of decisions fall back to
the LLM on dev.
**Deviations from plan:** gate on `answer_confidence` rather than entropy `confidence` (chosen on dev, higher
coverage at equal accuracy; the library's source documents it as the calibrated quantity). Temperatures are applied
post-hoc per exact option count because Laya's own bucket (3-5) is shared by all our nodes. The dev accuracy for
tau selection is teacher-forced (the gold prefix), not end-to-end.
**Open issues:** the English checkpoint's 512-token context leaves ~316 tokens for the state, so long tickets are
truncated head+tail.

