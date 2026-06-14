# CLAUDE.md

Guidance for Claude agents working in this repository.

## Project overview

Conversational tourist assistant for Tenerife (master's final project). It combines:

- **RAG-as-a-tool** over `data/TENERIFE.pdf`: the retrieval pipeline is exposed to
  the LLM as a callable tool (`search_tourist_guide`), not hardwired into the prompt.
- **`get_weather` function call**: a real external function call hitting Open-Meteo
  (with a simulated fallback) for current Tenerife weather.
- **Multiturn dialogue**: stateful chat that keeps conversation history.

Stack: **Google Gemini via LangChain**, **FAISS** vector store, **Streamlit** UI and a
Jupyter notebook for the master deliverable.

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
  `DEFAULT_PDF_PATH`, `DEFAULT_INDEX_DIR`), immutable `Settings` dataclass and
  `load_settings` (reads env vars).
- `core/rag.py` — `TouristGuideRAG`: loads/splits the PDF, builds the FAISS index,
  embeds, searches and formats retrieved context.
- `core/weather.py` — `get_weather` (Open-Meteo + simulated fallback), `WeatherError`.
- `core/tools.py` — wraps RAG and weather as LangChain tools: `search_tourist_guide`,
  `get_weather`, `get_tools`, plus `set_rag_instance` to inject the shared RAG.
- `core/assistant.py` — `TouristAssistant`: orchestrates the LLM, tool calls and
  history; `SYSTEM_PROMPT`, `chat`, `stream`, `reset`, `ToolCallRecord`.
- `app.py` — Streamlit UI (`main`, `get_settings`, `get_rag`, `render_sources`,
  `EXAMPLE_QUESTIONS`).
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

- The **FAISS index is built on first run** from `TENERIFE.pdf` and stored under
  `storage/`. It is **gitignored** (`storage/`, `*.faiss`, `index/`) and rebuilt
  locally — do not commit it.
- `.env` is gitignored; never commit credentials.
