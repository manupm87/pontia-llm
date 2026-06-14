# CLAUDE.md

Guidance for Claude agents working in this repository.

## Project overview

Conversational tourist assistant for Tenerife (master's final project). It combines:

- **RAG-as-a-tool** over `data/TENERIFE.pdf`: the retrieval pipeline is exposed to
  the LLM as a callable tool (`search_tourist_guide`), not hardwired into the prompt.
  Answers are **grounded only in the retrieved fragments** (the prompt forbids
  parametric knowledge) and the tool scaffolding is kept **out of the persistent
  history**, so every turn re-queries the guide and refreshes its citations.
- **Location photos**: the guide's embedded photos are extracted and mapped by page
  (`core/images.py`); photos are **interleaved inline** in the answer next to the
  place they depict (`core/photo_match.py`).
- **Function calls**: `get_weather` and `get_sea_conditions` hit Open-Meteo (with a
  simulated fallback); `resolve_date` turns relative Spanish dates into ISO.
- **Multiturn dialogue**: stateful chat that keeps conversation history.
- **Streaming + reasoning**: answer streams in the body; Gemini "thinking" streams
  live when enabled. Streaming uses an extra generation vs the one-shot path — the
  sidebar toggle lets users trade UX for cost.
- **Guardrails**: rule-based prompt-injection detection (always on) plus optional
  LLM topic/grounding guardrails (`core/guardrails.py`).
- **Evaluation**: an LLM-as-judge harness with metrics + report (`core/evaluation.py`,
  `scripts/run_eval.py`).
- **Observability**: per-turn and cumulative token usage + estimated cost.

Stack: **Google Gemini via LangChain**, **FAISS** vector store, **PyMuPDF** for image
extraction, **Streamlit** UI and a Jupyter notebook for the master deliverable.
Tests live in `tests/` and run with `python -m pytest` (no network/heavy deps:
the LLM and RAG are faked).

## Bilingual rule (mandatory)

The user insists on this split. Respect it in every file you touch:

- **Code in English**: all identifiers — functions, variables, classes, modules.
- **Documentation in Spanish**: docstrings, inline comments, notebook markdown,
  `README.md`, `INFORME.md`.
- **End-user-facing text in Spanish**: LLM system prompts, Streamlit labels, error
  messages (the assistant speaks Spanish to tourists).
- **`CLAUDE.md` in English** (this file).

## File / module map

The reusable package lives in `core/`:

- `core/config.py` — Tenerife coordinates, project paths (`PROJECT_ROOT`,
  `DEFAULT_PDF_PATH`, `DEFAULT_INDEX_DIR`, `DEFAULT_IMAGES_DIR`), immutable `Settings`
  dataclass (chunking `chunk_size=500`/`chunk_overlap=100`/`top_k=5`, `thinking_budget`,
  `images_dir`, `min_image_size`, `max_images_shown`) and `load_settings` (env vars
  incl. `THINKING_BUDGET`).
- `core/rag.py` — `TouristGuideRAG`: loads/splits the PDF, builds the FAISS index,
  embeds, searches and formats retrieved context; `retrieve` returns context + full
  source fragments + photos. Photos come via its `GuideImageStore`.
- `core/images.py` — `GuideImageStore`: extracts the PDF's embedded photos with
  PyMuPDF, persists them + a `manifest.json` under `storage/images/`, and maps each
  page to its photos (`images_for_pages`).
- `core/weather.py` — `get_weather` (Open-Meteo + simulated fallback), `WeatherError`.
- `core/sea.py` — `get_sea_conditions` (Open-Meteo Marine + simulated fallback),
  `SeaError`; mirrors `weather.py`.
- `core/dates.py` — `resolve_date`: relative Spanish expressions ("mañana", "este
  finde", "el lunes que viene", "en 3 días") → ISO `YYYY-MM-DD`.
- `core/tools.py` — wraps RAG/weather/sea/date as LangChain tools:
  `search_tourist_guide` (returns `content_and_artifact` with per-call citations),
  `get_weather`, `get_sea_conditions`, `resolve_date`, `get_tools`, `set_rag_instance`.
- `core/guardrails.py` — `Guardrails` (input: rule-based `detect_injection` +
  optional LLM topic classifier; output: optional LLM grounding judge),
  `GuardVerdict`, `build_llm_guardrails`, refusal messages.
- `core/photo_match.py` — pure logic that interleaves photos at their mention:
  `plan_inline_images`, `place_tokens`, `normalize_text`.
- `core/evaluation.py` — eval harness: `EvalCase`, `EvalResult`, `run_evaluation`,
  `is_refusal`, `retrieval_hit`, `score_correct`, `summarize`, `to_dataframe`,
  `plot_summary`, `default_dataset` (pandas/matplotlib are lazy-imported).
- `core/assistant.py` — `TouristAssistant`: orchestrates the LLM, tool calls and
  history. `build_system_prompt` (date-anchored, Spanish, expert persona),
  `prepare`/`stream_reasoning_and_answer`/`stream`/`answer`/`chat`/`reset`,
  `TurnContext`, `ToolCallRecord`, token tracking (`total_usage`, `estimate_cost`),
  input guardrail wiring. Tool request/result messages are ephemeral, so only
  user/assistant turns persist; tracks `last_sources`/`last_images`.
- `app.py` — Streamlit UI: hero, clickable examples, live tool status, inline photos,
  reasoning panel, sources, sidebar controls (params, streaming, thinking, guardrails),
  token/cost display, export. Helpers: `render_*`, `handle_turn`, `get_assistant`.
- `ui_theme.py` — dynamic CSS theme + animated hero (`inject_styles`, `render_hero`).
- `scripts/run_eval.py` — runs the evaluation end to end and writes a CSV + PNG report.
- `tests/` — pytest suite for the pure/orchestration logic (LLM and RAG faked).
- Jupyter notebook — the master deliverable that walks through the pipeline end to end.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the API key in `.env` (use `.env.template` as a template):

```
GOOGLE_API_KEY="..."
```

Then:

```bash
streamlit run app.py   # web UI
jupyter lab            # notebook deliverable
```

## Key conventions

Mirror the master's session style (`resources/Large Lenguage Models/` if in doubt):

- First line of every `.py`: `from __future__ import annotations`.
- Modern type hints with `|` (`str | None`), never `Optional`.
- `@dataclass(frozen=True)` for immutable configuration.
- PEP8 and KISS: simple, readable code; no deep nesting or needless indirection.
- Public modules, classes and functions get triple-quoted Spanish docstrings;
  inline comments in Spanish, used sparingly (let the code explain itself).

## Notes

- The **FAISS index and the extracted photos are built on first run** from
  `TENERIFE.pdf` and stored under `storage/` (`storage/faiss_index/`,
  `storage/images/`). All of `storage/` is **gitignored** and rebuilt locally — do
  not commit it.
- `.env` is gitignored; never commit credentials.
