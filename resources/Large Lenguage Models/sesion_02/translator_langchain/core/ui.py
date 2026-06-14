from __future__ import annotations

from html import escape

import streamlit as st

from core.languages import (
    default_source_index,
    default_target_index,
    get_by_label,
    source_labels,
    target_labels,
)
from core.langchain_translator import PROVIDER_MODEL_OPTIONS, Provider
from core.models import ChatTurn, PendingLanguageMismatch, Register, TranslationRequest


REGISTER_OPTIONS: dict[str, Register] = {
    "Neutral": "neutral",
    "Formal": "formal",
    "Informal": "informal",
    "Técnico": "technical",
    "Marketing": "marketing",
}


def configure_page() -> None:
    st.set_page_config(
        page_title="LangChain Translator",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_styles(), unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        """
        <section class="topbar">
            <div class="brand">
                <div class="brand-mark">AI</div>
                <div>
                    <h1>Traductor conversacional</h1>
                    <p class="subtitle">
                        Traduce texto entre idiomas manteniendo tono, formato y contexto lingüístico.
                    </p>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(
    default_provider: str,
    default_model: str,
) -> tuple[TranslationRequest | None, Provider, str]:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-heading">
                <h2>Configuración</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        source_label = st.selectbox(
            "Idioma origen",
            source_labels(),
            index=default_source_index(),
        )
        target_label = st.selectbox(
            "Idioma destino",
            target_labels(),
            index=default_target_index(),
        )
        register_label = st.selectbox("Registro", list(REGISTER_OPTIONS), index=0)
        preserve_formatting = st.toggle("Preservar formato", value=True)

        source_language = get_by_label(source_label)
        target_language = get_by_label(target_label)

        if source_language.code == target_language.code:
            st.warning("El idioma origen y destino son el mismo.")

        provider_options = list(PROVIDER_MODEL_OPTIONS)
        provider_index = (
            provider_options.index(default_provider)
            if default_provider in provider_options
            else 0
        )
        provider = st.selectbox(
            "Proveedor",
            provider_options,
            index=provider_index,
        )
        model_options = PROVIDER_MODEL_OPTIONS[provider]
        model_index = model_options.index(default_model) if default_model in model_options else 0
        model = st.selectbox(
            "Modelo",
            model_options,
            index=model_index,
        )

        st.markdown(
            f"""
            <div class="provider-footnote">
                <span>Proveedor y modelo</span>
                <strong>{escape(provider)} · {escape(model)}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    prompt = st.chat_input("Escribe el texto que quieres traducir")
    if not prompt:
        return None, provider, model

    request = TranslationRequest(
        text=prompt,
        source_language=source_language.name,
        target_language=target_language.name,
        source_language_display=source_language.display_name,
        target_language_display=target_language.display_name,
        register=REGISTER_OPTIONS[register_label],
        preserve_formatting=preserve_formatting,
    )
    return request, provider, model


def render_language_mismatch(pending: PendingLanguageMismatch) -> str | None:
    st.markdown(
        f"""
        <section class="language-warning">
            <div>
                <strong>Revisa el idioma de origen</strong>
                <p>
                    Has seleccionado <b>{escape(pending.request.source_language_display)}</b>,
                    pero el texto parece estar en
                    <b>{escape(pending.detected_source_language_display)}</b>.
                    ¿Quieres usar el idioma detectado?
                </p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    accept_col, reject_col, _ = st.columns([0.16, 0.18, 0.66])
    with accept_col:
        accept = st.button(
            f"Usar {pending.detected_source_language_display}",
            type="primary",
            use_container_width=True,
        )
    with reject_col:
        reject = st.button(
            f"Mantener {pending.request.source_language_display}",
            use_container_width=True,
        )

    if accept:
        return "accept"
    if reject:
        return "reject"
    return None


def render_history(turns: list[ChatTurn], visible_count: int) -> tuple[bool, bool]:
    if not turns:
        render_empty_state()
        return False, False

    clear_clicked = render_conversation_actions(turns)
    ordered_turns = list(reversed(turns))
    visible_turns = ordered_turns[:visible_count]
    has_more = len(ordered_turns) > len(visible_turns)

    for index, turn in enumerate(visible_turns):
        fade_class = _fade_class(index, len(visible_turns))
        st.markdown(
            f"""
            <article class="translation-pair {fade_class}">
                <section class="message-card source-card">
                    <div class="message-meta">
                        <span>{escape(_display_language(turn, "source"))}</span>
                    </div>
                    <div class="message-body">{_format_text(turn.source_text)}</div>
                </section>
                <section class="message-card target-card">
                    <div class="message-meta">
                        <span>{escape(_target_label_with_register(turn))}</span>
                        <span>{turn.created_at.strftime("%Y-%m-%d %H:%M")}</span>
                    </div>
                    <div class="message-body">{_format_text(turn.translation)}</div>
                </section>
            </article>
            """,
            unsafe_allow_html=True,
        )
        if turn.notes:
            with st.expander("Notas de traducción"):
                for note in turn.notes:
                    st.markdown(f"- {note}")

    show_more_clicked = False
    if has_more:
        st.markdown('<div class="show-more-marker">', unsafe_allow_html=True)
        show_more_clicked = st.button(
            "Mostrar más traducciones",
            type="tertiary",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    return clear_clicked, show_more_clicked


def render_streaming_turn(
    placeholder: st.delta_generator.DeltaGenerator,
    request: TranslationRequest,
    translation: str,
) -> None:
    placeholder.markdown(
        f"""
        <article class="translation-pair streaming-pair">
            <section class="message-card source-card">
                <div class="message-meta">
                    <span>{escape(request.source_language_display)}</span>
                </div>
                <div class="message-body">{_format_text(request.text)}</div>
            </section>
            <section class="message-card target-card">
                <div class="message-meta">
                    <span>{escape(_request_target_label_with_register(request))}</span>
                    <span>Generando...</span>
                </div>
                <div class="message-body">{_format_text(translation) if translation else '<span class="stream-cursor">▌</span>'}</div>
            </section>
        </article>
        """,
        unsafe_allow_html=True,
    )


def render_conversation_actions(turns: list[ChatTurn]) -> bool:
    st.markdown('<div class="conversation-toolbar">', unsafe_allow_html=True)
    _, clear_col, export_col = st.columns(
        [1, 0.032, 0.032],
        gap="small",
        vertical_alignment="center",
    )
    with clear_col:
        clear_clicked = st.button(
            "↺",
            help="Limpiar historial",
            type="tertiary",
        )
    with export_col:
        export_history(turns)
    st.markdown("</div>", unsafe_allow_html=True)
    return clear_clicked


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-mark">→</div>
            <h2>Introduce un texto para empezar</h2>
            <p>Selecciona idioma origen, idioma destino y registro en la barra lateral.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_missing_key(provider: str = "OpenAI") -> None:
    key_name = "OPENAI_API_KEY" if provider == "OpenAI" else "GOOGLE_API_KEY"
    st.error(f"No se ha encontrado `{key_name}`.")
    st.info("Crea un archivo `.env` a partir de `.env.template` o exporta la variable de entorno.")


def render_error(message: str) -> None:
    st.error(message)


def export_history(turns: list[ChatTurn]) -> None:
    data = _history_to_markdown(turns)
    st.download_button(
        "↓",
        data=data,
        file_name="translation_history.md",
        mime="text/markdown",
        help="Exportar historial",
        disabled=not turns,
        type="tertiary",
    )


def _history_to_markdown(turns: list[ChatTurn]) -> str:
    if not turns:
        return ""

    chunks = ["# Translation history", ""]
    for index, turn in enumerate(turns, start=1):
        chunks.extend(
            [
                f"## Translation {index}",
                "",
                f"- Source: {_display_language(turn, 'source')}",
                f"- Target: {_display_language(turn, 'target')}",
                f"- Register: {turn.register}",
                f"- Created at: {turn.created_at.isoformat(timespec='seconds')}",
                "",
                "### Source text",
                "",
                turn.source_text,
                "",
                "### Translation",
                "",
                turn.translation,
                "",
            ]
        )
        if turn.notes:
            chunks.extend(["### Notes", ""])
            chunks.extend([f"- {note}" for note in turn.notes])
            chunks.append("")
    return "\n".join(chunks)


def _format_text(value: str) -> str:
    return "<br>".join(escape(value).splitlines())


def _fade_class(index: int, total: int) -> str:
    if total < 5:
        return ""
    if index == total - 1:
        return "fade-strong"
    if index == total - 2:
        return "fade-soft"
    return ""


def _display_language(turn: ChatTurn, side: str) -> str:
    if side == "source":
        return getattr(turn, "source_language_display", None) or _to_spanish_language_name(
            turn.source_language
        )
    return getattr(turn, "target_language_display", None) or _to_spanish_language_name(
        turn.target_language
    )


def _target_label_with_register(turn: ChatTurn) -> str:
    return f'{_display_language(turn, "target")} ({_display_register(turn.register)})'


def _request_target_label_with_register(request: TranslationRequest) -> str:
    return f"{request.target_language_display} ({_display_register(request.register)})"


def _display_register(register: str) -> str:
    names = {
        "neutral": "neutral",
        "formal": "formal",
        "informal": "informal",
        "technical": "técnico",
        "marketing": "marketing",
    }
    return names.get(register, register)


def _to_spanish_language_name(language_name: str) -> str:
    names = {
        "English": "Inglés",
        "Spanish": "Español",
        "Italian": "Italiano",
        "French": "Francés",
        "German": "Alemán",
        "Portuguese": "Portugués",
        "Catalan": "Catalán",
        "Galician": "Gallego",
        "Basque": "Euskera",
        "Dutch": "Neerlandés",
        "Chinese": "Chino",
        "Japanese": "Japonés",
        "Korean": "Coreano",
        "Arabic": "Árabe",
    }
    return names.get(language_name, language_name)


def _styles() -> str:
    return """
    <style>
        :root {
            --ink: #050505;
            --muted: #5b625f;
            --paper: #ffffff;
            --line: #d9dedb;
            --navy: #0d1828;
            --orange: #f5660b;
            --green: #143f35;
            --yellow: #f7ffa8;
            --yellow-soft: #fbffd7;
            --orange-soft: #fff0e6;
            --green-soft: #e7f2ee;
            --navy-soft: #e9eef4;
        }

        html,
        body,
        [data-testid="stAppViewContainer"] {
            color: var(--ink);
            background:
                linear-gradient(180deg, rgba(251, 255, 215, 0.72) 0%, rgba(255, 255, 255, 0.92) 230px),
                var(--paper);
        }

        [data-testid="stAppViewContainer"] {
            height: 100vh;
            overflow: hidden;
        }

        [data-testid="stMain"] {
            height: 100vh;
            overflow: hidden;
        }

        .block-container {
            max-width: 1180px;
            height: 100vh;
            padding-top: 1.25rem;
            padding-bottom: 5.4rem;
            overflow: hidden;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(13, 24, 40, 0.98) 0%, rgba(20, 63, 53, 0.98) 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.12);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 2rem;
        }

        .topbar {
            z-index: 50;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
            margin-bottom: 6px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 18px 45px rgba(13, 24, 40, 0.08);
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.58);
            border-radius: 8px;
            scrollbar-color: rgba(13, 24, 40, 0.28) transparent;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar {
            width: 8px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar-thumb {
            background: rgba(13, 24, 40, 0.24);
            border-radius: 999px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.translation-pair),
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.empty-state),
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.language-warning) {
            height: calc(100vh - 275px) !important;
            min-height: 360px;
            max-height: 680px;
            padding: 4px 10px 82px 0;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            overscroll-behavior: contain;
            scrollbar-gutter: stable;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.translation-pair) > div,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.empty-state) > div,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.language-warning) > div {
            min-height: 100%;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 14px;
            min-width: 0;
        }

        .brand-mark {
            display: grid;
            width: 48px;
            height: 48px;
            flex: 0 0 48px;
            place-items: center;
            color: #ffffff;
            background: var(--navy);
            border-radius: 8px;
            box-shadow: 0 10px 26px rgba(13, 24, 40, 0.18);
            font-size: 13px;
            font-weight: 900;
        }

        .topbar h1 {
            color: var(--ink);
            font-size: 2rem;
            line-height: 1.15;
            margin: 0.05rem 0 0.35rem;
            letter-spacing: 0;
        }

        .eyebrow {
            color: var(--muted);
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0;
            margin: 0;
            text-transform: uppercase;
        }

        .subtitle {
            color: var(--muted);
            font-size: 1rem;
            margin: 0;
            max-width: 760px;
        }

        .sidebar-heading {
            margin-bottom: 18px;
        }

        .sidebar-heading h2 {
            margin: 0;
            color: #ffffff;
            font-size: 1.25rem;
            line-height: 1.2;
        }

        [data-testid="stSidebar"] div[data-testid="stSelectbox"] {
            margin-bottom: 10px;
        }

        [data-testid="stSidebar"] div[data-testid="stSelectbox"] label,
        [data-testid="stSidebar"] div[data-testid="stToggle"] label {
            color: rgba(255, 255, 255, 0.68);
            font-size: 0.82rem;
            font-weight: 800;
        }

        [data-testid="stSidebar"] div[data-testid="stSelectbox"] label p,
        [data-testid="stSidebar"] div[data-testid="stToggle"] label p {
            color: rgba(255, 255, 255, 0.68);
            font-size: 0.82rem;
            font-weight: 800;
        }

        [data-testid="stSidebar"] div[data-testid="stToggle"] label,
        [data-testid="stSidebar"] div[data-testid="stToggle"] label p {
            color: #ffffff !important;
        }

        [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] span {
            color: #ffffff !important;
        }

        [data-testid="stSidebar"] [data-testid="stToggle"] p,
        [data-testid="stSidebar"] [data-testid="stToggle"] span,
        [data-testid="stSidebar"] [data-testid="stToggle"] div {
            color: #ffffff !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            min-height: 46px;
            color: #ffffff;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 8px;
            box-shadow: none;
            transition:
                background 160ms ease,
                border-color 160ms ease,
                transform 160ms ease;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div:hover {
            background: rgba(255, 255, 255, 0.12);
            border-color: rgba(245, 102, 11, 0.72);
            transform: translateY(-1px);
        }

        [data-testid="stSidebar"] [data-baseweb="select"] div,
        [data-testid="stSidebar"] [data-baseweb="select"] span {
            color: #ffffff;
            font-weight: 700;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] svg {
            fill: rgba(255, 255, 255, 0.72);
        }

        [data-testid="stSidebar"] [data-testid="stToggle"] {
            margin-top: 10px;
        }

        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.16);
            margin: 26px 0 18px;
        }

        .provider-footnote {
            position: fixed;
            bottom: 18px;
            left: 18px;
            width: calc(21rem - 36px);
            color: rgba(255, 255, 255, 0.54);
            font-size: 0.72rem;
            line-height: 1.35;
        }

        .provider-footnote span {
            display: block;
            margin-bottom: 2px;
        }

        .provider-footnote strong {
            display: block;
            color: rgba(255, 255, 255, 0.76);
            font-size: 0.75rem;
            letter-spacing: 0;
        }

        .conversation-toolbar {
            margin: -2px 0 0;
            height: 0;
        }

        div[data-testid="stMarkdownContainer"]:has(.conversation-toolbar) {
            height: 0 !important;
            margin: 0 !important;
        }

        div[data-testid="stMarkdownContainer"]:has(.conversation-toolbar) + div[data-testid="stHorizontalBlock"] {
            margin-top: -8px !important;
            margin-bottom: 2px !important;
        }

        div[data-testid="stMarkdownContainer"]:has(.conversation-toolbar) + div[data-testid="stHorizontalBlock"] [data-testid="stButton"] button,
        div[data-testid="stMarkdownContainer"]:has(.conversation-toolbar) + div[data-testid="stHorizontalBlock"] [data-testid="stDownloadButton"] button {
            width: 28px !important;
            min-width: 28px !important;
            height: 28px !important;
            min-height: 28px !important;
            padding: 0 !important;
            color: var(--navy) !important;
            background: transparent !important;
            border: 0 !important;
            outline: 0 !important;
            box-shadow: none !important;
            border-radius: 0 !important;
            font-size: 1.2rem !important;
            font-weight: 900 !important;
            line-height: 1 !important;
        }

        div[data-testid="stMarkdownContainer"]:has(.conversation-toolbar) + div[data-testid="stHorizontalBlock"] [data-testid="stButton"] button:hover,
        div[data-testid="stMarkdownContainer"]:has(.conversation-toolbar) + div[data-testid="stHorizontalBlock"] [data-testid="stDownloadButton"] button:hover {
            color: var(--orange) !important;
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            transform: translateY(-1px);
        }

        div[data-testid="stMarkdownContainer"]:has(.conversation-toolbar) + div[data-testid="stHorizontalBlock"] [data-testid="stButton"] button:focus,
        div[data-testid="stMarkdownContainer"]:has(.conversation-toolbar) + div[data-testid="stHorizontalBlock"] [data-testid="stButton"] button:active,
        div[data-testid="stMarkdownContainer"]:has(.conversation-toolbar) + div[data-testid="stHorizontalBlock"] [data-testid="stDownloadButton"] button:focus,
        div[data-testid="stMarkdownContainer"]:has(.conversation-toolbar) + div[data-testid="stHorizontalBlock"] [data-testid="stDownloadButton"] button:active {
            color: var(--orange) !important;
            background: transparent !important;
            border: 0 !important;
            outline: 0 !important;
            box-shadow: none !important;
        }

        .language-warning {
            margin: 0 0 10px;
            padding: 14px 16px;
            color: var(--ink);
            background: var(--orange-soft);
            border: 1px solid #f7b17f;
            border-radius: 8px;
            box-shadow: 0 14px 32px rgba(13, 24, 40, 0.06);
        }

        .language-warning strong {
            display: block;
            margin-bottom: 4px;
            color: var(--navy);
            font-size: 0.95rem;
        }

        .language-warning p {
            margin: 0;
            color: var(--muted);
            line-height: 1.45;
        }

        .translation-pair {
            display: grid;
            grid-template-columns: minmax(0, 0.92fr) minmax(0, 1.08fr);
            gap: 12px;
            margin: -4px 0 14px;
        }

        .translation-pair.fade-soft {
            opacity: 0.58;
        }

        .translation-pair.fade-strong {
            opacity: 0.32;
        }

        .translation-pair.fade-soft,
        .translation-pair.fade-strong {
            filter: saturate(0.72);
        }

        .streaming-pair .target-card {
            box-shadow:
                0 14px 32px rgba(13, 24, 40, 0.07),
                inset 0 0 0 1px rgba(247, 255, 168, 0.28);
        }

        .stream-cursor {
            display: inline-block;
            color: rgba(255, 255, 255, 0.78);
            animation: blink 900ms steps(2, start) infinite;
        }

        @keyframes blink {
            0%, 45% { opacity: 1; }
            46%, 100% { opacity: 0; }
        }

        .show-more-marker {
            margin: 4px 0 16px;
            text-align: center;
        }

        div[data-testid="stMarkdownContainer"]:has(.show-more-marker) + div[data-testid="stButton"] {
            display: flex;
            justify-content: center;
        }

        div[data-testid="stMarkdownContainer"]:has(.show-more-marker) + div[data-testid="stButton"] button {
            width: auto !important;
            min-width: 0 !important;
            height: auto !important;
            min-height: 0 !important;
            padding: 0 !important;
            color: var(--navy) !important;
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            font-size: 0.88rem !important;
            font-weight: 800 !important;
            text-decoration: underline;
            text-underline-offset: 3px;
        }

        div[data-testid="stMarkdownContainer"]:has(.show-more-marker) + div[data-testid="stButton"] button:hover {
            color: var(--orange) !important;
            background: transparent !important;
        }

        .message-card {
            min-width: 0;
            padding: 15px;
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 14px 32px rgba(13, 24, 40, 0.07);
        }

        .source-card {
            background: var(--yellow-soft);
            border-color: #eaf276;
        }

        .target-card {
            color: #ffffff;
            background: var(--green);
            border-color: var(--green);
        }

        .message-meta {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            margin-bottom: 10px;
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 900;
            letter-spacing: 0;
            text-transform: uppercase;
        }

        .target-card .message-meta {
            color: rgba(255, 255, 255, 0.72);
        }

        .message-body {
            color: inherit;
            font-size: 1rem;
            line-height: 1.55;
            overflow-wrap: anywhere;
        }

        .source-card .message-body {
            color: var(--ink);
        }

        .stExpander {
            border-color: var(--line);
        }

        .empty-state {
            display: grid;
            place-items: center;
            min-height: 280px;
            background: rgba(255, 255, 255, 0.96);
            border: 1px dashed #b9d7cd;
            border-radius: 8px;
            margin-top: 1rem;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 18px 45px rgba(13, 24, 40, 0.06);
        }

        .empty-mark {
            display: grid;
            width: 52px;
            height: 52px;
            place-items: center;
            margin-bottom: 12px;
            color: #ffffff;
            background: var(--orange);
            border-radius: 8px;
            font-size: 1.45rem;
            font-weight: 900;
        }

        .empty-state h2 {
            color: var(--ink);
            font-size: 1.25rem;
            margin-bottom: 0.35rem;
        }

        .empty-state p {
            color: var(--muted);
            margin-bottom: 0;
        }

        [data-testid="stChatInput"] {
            background: rgba(255, 255, 255, 0.92);
            border-top: 1px solid var(--line);
        }

        [data-testid="stChatInput"] textarea {
            border: 1px solid var(--line);
            border-radius: 8px;
        }

        @media (max-width: 820px) {
            .topbar {
                padding: 14px;
            }

            .brand {
                align-items: flex-start;
            }

            .topbar h1 {
                font-size: 1.55rem;
            }

            .translation-pair {
                grid-template-columns: 1fr;
            }

            .provider-footnote {
                position: static;
                width: auto;
                margin-top: 28px;
            }
        }
    </style>
    """
