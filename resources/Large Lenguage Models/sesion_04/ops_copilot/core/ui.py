from __future__ import annotations

import json
from html import escape
from typing import Any

import pandas as pd
import streamlit as st

from core.models import ChatTurn, ExpenseDraft
from core.tools import TOOL_GROUPS


CONFIDENCE_LABELS = {
    "high": "Alta",
    "medium": "Media",
    "low": "Baja",
}


def configure_page() -> None:
    st.set_page_config(
        page_title="Ops Copilot",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_styles(), unsafe_allow_html=True)


def render_header(stats: dict[str, Any]) -> None:
    st.markdown(
        f"""
        <section class="topbar">
            <div class="brand">
                <div class="brand-mark">OP</div>
                <div>
                    <h1>Ops Copilot</h1>
                    <p class="subtitle">
                        Asistente operativo con herramientas auditables para políticas internas,
                        inventario, métricas comerciales y borradores de gasto.
                    </p>
                </div>
            </div>
        </section>
        <section class="status-strip">
            <div><span>Revenue 2024</span><strong>{_eur(stats["total_revenue"])}</strong></div>
            <div><span>Margen 2024</span><strong>{_eur(stats["total_margin"])}</strong></div>
            <div><span>Usuarios activos</span><strong>{stats["latest_active_users"]:,}</strong></div>
            <div><span>NPS medio</span><strong>{stats["average_nps"]}</strong></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_missing_key() -> None:
    st.error("No se ha encontrado `OPENAI_API_KEY`.")
    st.info(
        "Crea un archivo `.env` en esta carpeta o exporta la variable de entorno antes de lanzar la aplicación."
    )


def render_error(message: str) -> None:
    st.error(message)


def render_sidebar(default_top_k: int, model: str) -> dict[str, Any]:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-heading">
                <h2>Configuración</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        enabled_groups = _render_tool_selector()
        tool_mode_label = st.radio(
            "Selección de herramientas",
            ["Automática", "Obligatoria", "Desactivada"],
            index=0,
            horizontal=False,
        )
        top_k = st.slider(
            "Evidencias documentales",
            min_value=1,
            max_value=8,
            value=default_top_k,
            help="Número recomendado de fragmentos cuando el asistente busca documentación interna.",
        )
        parallel_tool_calls = st.toggle(
            "Llamadas paralelas",
            value=True,
            help="Permite que el modelo solicite varias herramientas de lectura en una misma ronda.",
        )
        st.markdown(
            f"""
            <div class="provider-footnote">
                <span>Proveedor</span>
                <strong>OpenAI · {escape(model)}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return {
        "enabled_groups": set(enabled_groups),
        "tool_choice": {
            "Automática": "auto",
            "Obligatoria": "required",
            "Desactivada": "none",
        }[tool_mode_label],
        "top_k": top_k,
        "parallel_tool_calls": parallel_tool_calls,
    }


def render_suggested_prompts() -> str | None:
    prompts = [
        "Tengo una factura de 720 EUR de DataCloud por licencias anuales. Prepara el gasto y dime qué aprobaciones hacen falta.",
        "¿Qué región tiene más revenue y cuál tiene mejor margen?",
        "¿Hay stock del monitor-27 y del teclado-mecanico?",
        "¿Cómo debo comunicar una incidencia SEV2 a un cliente?",
    ]
    st.markdown('<section class="prompt-bank">', unsafe_allow_html=True)
    cols = st.columns(len(prompts), gap="small")
    selected = None
    for idx, prompt in enumerate(prompts):
        with cols[idx]:
            if st.button(prompt, use_container_width=True, key=f"suggested_{idx}"):
                selected = prompt
    st.markdown("</section>", unsafe_allow_html=True)
    return selected


def render_history(turns: list[ChatTurn], visible_count: int) -> tuple[bool, bool]:
    if not turns:
        return False, False

    clear_clicked = _render_conversation_actions()
    ordered_turns = list(reversed(turns))
    visible_turns = ordered_turns[:visible_count]

    for turn in visible_turns:
        _render_turn(turn)

    show_more_clicked = False
    if len(ordered_turns) > len(visible_turns):
        show_more_clicked = st.button("Mostrar más", type="secondary")
    return clear_clicked, show_more_clicked


def render_trace_panel(turns: list[ChatTurn]) -> None:
    if not turns:
        st.info("Todavía no hay trazas. Ejecuta una consulta desde el copilot.")
        return

    for turn in reversed(turns):
        st.markdown(
            f"#### {escape(turn.question)}\n"
            f"{len(turn.executions)} llamadas · {turn.created_at.strftime('%Y-%m-%d %H:%M')}"
        )
        if not turn.executions:
            st.caption("El modelo respondió sin herramientas.")
            continue
        for execution in turn.executions:
            status = "OK" if execution.ok else "ERROR"
            with st.expander(
                f"{execution.name} · {status} · {execution.elapsed_seconds:.2f}s",
                expanded=False,
            ):
                st.markdown("**Argumentos**")
                st.json(execution.arguments)
                st.markdown("**Salida**")
                st.json(_json_safe(execution.output))


def render_drafts_panel(draft_store: dict[str, ExpenseDraft]) -> tuple[str | None, str | None]:
    if not draft_store:
        st.info("No hay borradores pendientes.")
        return None, None

    action_draft_id = None
    action_status = None
    for draft in reversed(list(draft_store.values())):
        st.markdown(
            f"""
            <article class="draft-row {escape(draft.status)}">
                <div>
                    <span>{escape(draft.draft_id)}</span>
                    <h3>{escape(draft.vendor)} · {_eur(draft.amount_eur)}</h3>
                    <p>{escape(draft.concept)} · Centro de coste: {escape(draft.cost_center)}</p>
                </div>
                <strong>{escape(_draft_status_label(draft.status))}</strong>
            </article>
            """,
            unsafe_allow_html=True,
        )
        if draft.status == "requires_human_confirmation":
            col_a, col_b, _ = st.columns([0.13, 0.13, 0.74])
            with col_a:
                if st.button("Confirmar", key=f"confirm_{draft.draft_id}", type="primary"):
                    action_draft_id = draft.draft_id
                    action_status = "confirmed"
            with col_b:
                if st.button("Rechazar", key=f"reject_{draft.draft_id}"):
                    action_draft_id = draft.draft_id
                    action_status = "rejected"
    return action_draft_id, action_status


def render_data_panel(metrics: pd.DataFrame, inventory: pd.DataFrame, knowledge_stats: dict[str, int]) -> None:
    left, right = st.columns([0.62, 0.38], gap="large")
    with left:
        revenue = metrics.pivot(index="date", columns="region", values="revenue")
        st.markdown("#### Revenue mensual por región")
        st.line_chart(revenue)
        st.markdown("#### Datos comerciales")
        st.dataframe(metrics, use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Inventario")
        st.dataframe(inventory, use_container_width=True, hide_index=True)
        st.markdown(
            f"""
            <section class="knowledge-panel">
                <span>Base documental</span>
                <strong>{knowledge_stats["sources"]} fuentes · {knowledge_stats["chunks"]} fragmentos</strong>
            </section>
            """,
            unsafe_allow_html=True,
        )


def _render_conversation_actions() -> bool:
    _, clear_col = st.columns([0.9, 0.1])
    with clear_col:
        return st.button("Limpiar", type="secondary", use_container_width=True)


def _render_tool_selector() -> list[str]:
    options = list(TOOL_GROUPS)
    selected = []
    with st.popover("Herramientas disponibles", use_container_width=True):
        st.caption("Activa solo las capacidades disponibles para la siguiente consulta.")
        for option in options:
            enabled = st.checkbox(option, value=True, key=f"tool_group_{option}")
            if enabled:
                selected.append(option)
    st.caption(f"{len(selected)} de {len(options)} herramientas activas")
    return selected


def _render_turn(turn: ChatTurn) -> None:
    tools = " ".join(_chip(tool) for tool in turn.answer.used_tools) or _chip("sin herramientas")
    sources = " ".join(_chip(source) for source in turn.answer.sources) or _chip("sin fuentes")
    next_step = (
        f'<div class="next-step"><span>Siguiente paso</span><p>{escape(turn.answer.next_step)}</p></div>'
        if turn.answer.next_step
        else ""
    )
    st.markdown(
        f"""
        <article class="turn-card">
            <div class="question">{escape(turn.question)}</div>
            <div class="answer">{_format_text(turn.answer.answer)}</div>
            <div class="turn-meta">
                <span>Confianza: {CONFIDENCE_LABELS.get(turn.answer.confidence, turn.answer.confidence)}</span>
                <span>{turn.created_at.strftime('%Y-%m-%d %H:%M')}</span>
            </div>
            <div class="chip-row">{tools}</div>
            <div class="source-row">{sources}</div>
            {next_step}
        </article>
        """,
        unsafe_allow_html=True,
    )


def _chip(label: str) -> str:
    return f'<span class="chip">{escape(label)}</span>'


def _format_text(text: str) -> str:
    paragraphs = [escape(part.strip()) for part in text.split("\n") if part.strip()]
    return "".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)


def _eur(value: float) -> str:
    return f"{value:,.0f} EUR".replace(",", ".")


def _draft_status_label(status: str) -> str:
    return {
        "requires_human_confirmation": "Pendiente",
        "confirmed": "Confirmado",
        "rejected": "Rechazado",
    }.get(status, status)


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str, ensure_ascii=False))


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

        .stApp {
            color: var(--ink);
            background:
                linear-gradient(180deg, rgba(251, 255, 215, 0.70) 0%, rgba(255, 255, 255, 0.94) 260px),
                var(--paper);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(13, 24, 40, 0.98) 0%, rgba(20, 63, 53, 0.98) 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.12);
        }

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] h2 {
            color: #ffffff !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="tag"] {
            background: rgba(255, 255, 255, 0.10);
            border-color: rgba(255, 255, 255, 0.16);
            color: #ffffff;
        }

        [data-testid="stSidebar"] [data-testid="stSlider"] * {
            color: #ffffff !important;
        }

        [data-testid="stSidebar"] div[data-testid="stTickBar"] *,
        [data-testid="stSidebar"] div[data-testid="stTickBarMin"],
        [data-testid="stSidebar"] div[data-testid="stTickBarMax"] {
            color: #ffffff !important;
        }

        [data-testid="stSidebar"] div[data-testid="stPopover"] button {
            color: #ffffff !important;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 8px;
        }

        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1.5rem;
            padding: 1.05rem 0 1rem;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .brand-mark {
            display: grid;
            place-items: center;
            width: 54px;
            height: 54px;
            border-radius: 8px;
            color: #ffffff;
            background: var(--navy);
            font-weight: 900;
            letter-spacing: 0;
            box-shadow: 0 10px 26px rgba(13, 24, 40, 0.18);
        }

        .topbar h1 {
            margin: 0;
            font-size: clamp(2rem, 3vw, 3.2rem);
            letter-spacing: 0;
            color: var(--ink);
        }

        .subtitle {
            margin: 0.2rem 0 0;
            max-width: 820px;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.55;
        }

        .knowledge-panel {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
            background: rgba(255, 255, 255, 0.82);
            min-width: 170px;
        }

        .knowledge-panel span,
        .status-strip span {
            display: block;
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .knowledge-panel strong {
            color: var(--green);
            font-size: 0.92rem;
        }

        .status-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0 0 1rem;
        }

        .status-strip div {
            padding: 0.9rem 1rem;
            border: 1px solid var(--line);
            border-left: 4px solid var(--orange);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.9);
            box-shadow: 0 10px 24px rgba(13, 24, 40, 0.05);
        }

        .status-strip strong {
            display: block;
            margin-top: 0.15rem;
            color: var(--ink);
            font-size: 1.25rem;
        }

        .prompt-bank {
            margin: 0.25rem 0 0.8rem;
        }

        .turn-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.96);
            padding: 1.05rem 1.1rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 12px 26px rgba(13, 24, 40, 0.06);
        }

        .question {
            color: var(--green);
            font-weight: 800;
            margin-bottom: 0.75rem;
        }

        .answer p {
            color: var(--ink);
            line-height: 1.58;
            margin: 0 0 0.65rem;
        }

        .turn-meta {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            color: var(--muted);
            font-size: 0.84rem;
            padding-top: 0.2rem;
        }

        .chip-row,
        .source-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin-top: 0.55rem;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            min-height: 26px;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: var(--navy-soft);
            color: var(--navy);
            font-size: 0.8rem;
        }

        .source-row .chip {
            background: var(--green-soft);
            color: var(--green);
        }

        .next-step {
            margin-top: 0.85rem;
            padding-top: 0.8rem;
            border-top: 1px dashed rgba(13, 24, 40, 0.16);
        }

        .next-step span {
            color: var(--orange);
            font-weight: 800;
            font-size: 0.85rem;
        }

        .next-step p {
            color: var(--muted);
            margin: 0.15rem 0 0;
        }

        .draft-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            border: 1px solid var(--line);
            border-left: 4px solid var(--orange);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.96);
            padding: 0.95rem 1rem;
            margin: 0.8rem 0 0.45rem;
        }

        .draft-row.confirmed {
            border-left-color: var(--green);
        }

        .draft-row.rejected {
            border-left-color: var(--muted);
        }

        .draft-row span {
            color: var(--muted);
            font-size: 0.8rem;
        }

        .draft-row h3 {
            margin: 0.15rem 0;
            color: var(--ink);
            font-size: 1rem;
        }

        .draft-row p {
            margin: 0;
            color: var(--muted);
        }

        .draft-row strong {
            color: var(--green);
        }

        .provider-footnote {
            margin-top: 1.25rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(255, 255, 255, 0.18);
            color: rgba(255, 255, 255, 0.78);
            font-size: 0.85rem;
        }

        .provider-footnote span {
            display: block;
            color: rgba(255, 255, 255, 0.62) !important;
        }

        div.stButton > button[kind="primary"],
        div.stButton > button[kind="primary"]:hover {
            background: var(--orange) !important;
            border-color: var(--orange) !important;
            color: #ffffff !important;
            border-radius: 8px;
        }

        div.stButton > button {
            border-radius: 8px;
        }

        @media (max-width: 900px) {
            .topbar,
            .draft-row {
                align-items: flex-start;
                flex-direction: column;
            }

            .status-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
    </style>
    """
