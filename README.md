# Hybrid System-1 / System-2 agent on a Procedural Graph (RefundDesk PoC)

This PoC routes a customer-support agent through an explicit **Procedural Graph** (`graphs/refunddesk_v1.yaml`):

- **System 1 – [Laya](https://github.com/NandhaKishorM/laya)**, a non-autoregressive typed-decision encoder, picks edges and runs the guard and the final reply review in a single forward pass.
- **System 2 – a local Qwen3.5-9B** (Ollama, Q4_K_M, text-only) is called only inside nodes that need generation or tool use, and as a fallback router when Laya's confidence is below `tau`.

Everything runs locally for $0. All measured results, per phase and with every deviation from the original plan, are in `RESULTS.md`.

## Setup

```bash
uv sync                                                  # Python 3.11 env (.venv)
ollama pull hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_M         # then build the text-only model:
#   edit FROM in ollama/Modelfile.qwen3.5-9b-text to the pulled model blob (ollama show --modelfile ...)
ollama create qwen3.5-9b-text -f ollama/Modelfile.qwen3.5-9b-text
uv run python scripts/vram_probe.py                      # both models on an 8 GB GPU?
```

### Fine-tuned Laya checkpoint (not included)

The fine-tuned checkpoint (`models/laya_refunddesk/`, 808 MB) is too large for git and is not in this repo. You need it only for E5, `hpg run --finetuned`, and the web UI's default chat mode ("Hybrid · fine-tuned Laya"). Everything else runs without it, and the stock Laya checkpoints download automatically on first use.

To recreate it (free, about 10 minutes of training):

1. `uv run python scripts/make_finetune_data.py` writes `data/finetune/refunddesk_routing_train.jsonl` (already committed, so this step is optional).
2. On [Kaggle](https://www.kaggle.com/), upload that file as a dataset, open `notebooks/refunddesk_finetune_2xT4_kaggle.ipynb`, attach the dataset, set the accelerator to **GPU T4 x2**, and run all cells.
3. Download `/kaggle/working/laya_refunddesk.zip` and unzip it so that `models/laya_refunddesk/model.safetensors` exists.
4. Calibrate it on dev: `uv run hpg calibrate --stage finetuned --device cpu`. This writes the `finetuned_calibration` block in `config.yaml`; the committed values are the ones used for the reported E5 results.

Your checkpoint will differ slightly from ours (training is stochastic), so E5 numbers may not match `RESULTS.md` exactly.

## Usage

```bash
uv run pytest -q                                         # unit + engine integration tests (no models needed)
uv run python scripts/validate_dataset.py                # gold paths valid, >= 8 gold examples per edge
uv run hpg run --ticket T0013 --router hybrid            # one ticket, per-step router + confidence
uv run hpg run --finetuned -i                           # interactive: type tickets, hybrid + fine-tuned Laya (E5)
uv run hpg run --finetuned --text "charged twice for ORD-10033"   # one custom ticket
uv run hpg calibrate --split dev                         # Phase 5: temperatures, checkpoint, gate, tau (dev only)
uv run hpg eval --all                                    # Phase 6: E1-E4 on test + reports/results_test.md
```

Traces are written to `runs/<experiment>/<ticket_id>.jsonl`, one JSON line per step. Each step records the node, the router that decided it (`deterministic | laya | llm | llm_fallback`), probabilities, confidence, latency, and LLM tokens. Metrics are computed from these traces only.

## Web UI

```bash
uv run hpg serve            # http://127.0.0.1:8000
```

- **Chat:** each message is one support ticket, run by the hybrid agent with the fine-tuned Laya by default (E3 zero-shot and LLM-only are selectable). The Procedural Graph lights up live, and every step shows who decided it (System 1 Laya / System 2 LLM), with confidence, latency, tool results and LLM tokens. Models load on the first message; **Unload models** frees the GPU.
- **PoC results:** KPIs, E1–E5 accuracy / latency / LLM cost / fallback charts, per-node accuracy, calibration, and the failure breakdown, computed from `reports/` and the test traces.

## Layout

| path | what |
| --- | --- |
| `graphs/refunddesk_v1.yaml` | the Procedural Graph: nodes, and edges with condition / guidance / pitfalls |
| `data/` | mock world (orders, KB, service status), 220 tickets (dev 70 / test 150), fine-tune data |
| `src/hpg/engine.py` | graph walker (max-steps guard, bounded retry edge) |
| `src/hpg/routers/` | `LLMRouter`, `LayaRouter`, `HybridRouter` behind one interface |
| `src/hpg/llm_nodes.py` | LLM nodes with PG guidance injection; tool loop for technical tickets |
| `src/hpg/eval/` | trace metrics, calibration (Phase 5), reports |
| `src/hpg/web/` | FastAPI server + single-page UI (chatbot and results dashboard) |
| `scripts/` | world / ticket / fine-tune data generators, dataset validator, VRAM probe |
| `notebooks/refunddesk_finetune_2xT4_kaggle.ipynb` | Phase 7 fine-tune on free Kaggle 2×T4 |
| `.claude/` | Claude Code agents, skills, and hooks used to build this repo |

## Follow-ups (not used here)

Laya ships a LangGraph integration (`laya.integrations.langchain.LayaRouter`). The core engine deliberately avoids agent frameworks so the System-1/System-2 handoff stays explicit and inspectable. Porting the graph to LangGraph conditional edges would be a natural next step.
