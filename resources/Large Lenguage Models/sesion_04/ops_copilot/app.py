from __future__ import annotations

import streamlit as st

from core.assistant import OpsAssistant
from core.config import DATA_DIR, Settings, load_settings
from core.data_sources import BusinessDataset
from core.state import (
    add_turn,
    clear_history,
    drafts,
    history,
    init_state,
    show_more_history,
    update_draft_status,
    visible_history_count,
)
from core.ui import (
    configure_page,
    render_data_panel,
    render_drafts_panel,
    render_error,
    render_header,
    render_history,
    render_missing_key,
    render_sidebar,
    render_suggested_prompts,
    render_trace_panel,
)


ASSISTANT_CACHE_VERSION = 1


@st.cache_resource(show_spinner=False)
def get_assistant(settings: Settings, cache_version: int) -> OpsAssistant:
    return OpsAssistant(
        api_key=settings.openai_api_key or "",
        generation_model=settings.generation_model,
        embedding_model=settings.embedding_model,
        timeout=settings.request_timeout,
        data_dir=DATA_DIR,
    )


def main() -> None:
    configure_page()
    init_state()

    settings = load_settings()
    render_header(
        BusinessDataset(DATA_DIR / "business_metrics.csv").overview(),
    )
    if not settings.has_api_key:
        render_missing_key()
        return

    assistant = get_assistant(settings, ASSISTANT_CACHE_VERSION)
    inputs = render_sidebar(settings.default_top_k, settings.generation_model)

    copilot_tab, traces_tab, drafts_tab, data_tab = st.tabs(
        ["Copilot", "Trazas", "Borradores", "Datos"]
    )

    with copilot_tab:
        selected_prompt = render_suggested_prompts()
        prompt = selected_prompt or st.chat_input("Escribe una solicitud operativa")
        _render_copilot(prompt, assistant, inputs)

    with traces_tab:
        render_trace_panel(history())

    with drafts_tab:
        draft_id, status = render_drafts_panel(drafts())
        if draft_id and status:
            update_draft_status(draft_id, status)
            st.rerun()

    with data_tab:
        render_data_panel(
            assistant.business.dataframe(),
            assistant.inventory.dataframe(),
            assistant.knowledge_stats,
        )


def _render_copilot(prompt: str | None, assistant: OpsAssistant, inputs) -> None:
    if prompt:
        with st.spinner("Ejecutando herramientas y preparando respuesta..."):
            try:
                turn = assistant.answer(
                    prompt,
                    history=history(),
                    enabled_groups=inputs["enabled_groups"],
                    draft_store=drafts(),
                    top_k=inputs["top_k"],
                    tool_choice=inputs["tool_choice"],
                    parallel_tool_calls=inputs["parallel_tool_calls"],
                )
            except Exception as exc:
                render_error(f"{type(exc).__name__}: {exc}")
                return
            add_turn(turn)
        st.rerun()

    with st.container(height=560, border=False):
        clear_clicked, show_more_clicked = render_history(history(), visible_history_count())

    if clear_clicked:
        clear_history()
        st.rerun()
    if show_more_clicked:
        show_more_history()
        st.rerun()


if __name__ == "__main__":
    main()
