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
  (`core/images.py`); the photos of the retrieved pages are shown next to the answer.
- **`get_weather` function call**: a real external function call hitting Open-Meteo
  (with a simulated fallback) for current Tenerife weather.
- **Multiturn dialogue**: stateful chat that keeps conversation history.

Stack: **Google Gemini via LangChain**, **FAISS** vector store, **PyMuPDF** for image
extraction, **Streamlit** UI and a Jupyter notebook for the master deliverable.

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
  dataclass (incl. `images_dir`, `min_image_size`, `max_images_shown`) and
  `load_settings` (reads env vars).
- `core/rag.py` — `TouristGuideRAG`: loads/splits the PDF, builds the FAISS index,
  embeds, searches and formats retrieved context; also exposes `last_images` and the
  page→photo lookup via its `GuideImageStore`.
- `core/images.py` — `GuideImageStore`: extracts the PDF's embedded photos with
  PyMuPDF, persists them + a `manifest.json` under `storage/images/`, and maps each
  page to its photos (`images_for_pages`).
- `core/weather.py` — `get_weather` (Open-Meteo + simulated fallback), `WeatherError`.
- `core/tools.py` — wraps RAG and weather as LangChain tools: `search_tourist_guide`,
  `get_weather`, `get_tools`, plus `set_rag_instance` to inject the shared RAG.
- `core/assistant.py` — `TouristAssistant`: orchestrates the LLM, tool calls and
  history; `SYSTEM_PROMPT`, `chat`, `stream`, `reset`, `ToolCallRecord`. Tool
  request/result messages are ephemeral (a working copy), so only the user/assistant
  turns persist; tracks `last_sources` and `last_images`.
- `app.py` — Streamlit UI (`main`, `get_settings`, `get_rag`, `render_sources`,
  `render_images`, `EXAMPLE_QUESTIONS`).
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
