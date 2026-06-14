from __future__ import annotations

import base64
import re
from html import escape
from pathlib import Path
import streamlit as st

from core.categories import (
    CATEGORIES,
    Category,
    default_index,
    get_by_label,
    get_by_name,
    labels as category_labels,
)
from core.models import (
    AnalyzedQuery,
    ChatTurn,
    Citation,
    IngestionConfig,
    RetrievalConfig,
    RetrievedChunk,
)
from core.rag_engine import IndexHandle
from core.use_cases import EvaluationResult, SuggestedCase


SIDEBAR_SECTION_TITLES = {
    "technical": "Ajustes técnicos",
}
LOGO_PATH = Path(__file__).resolve().parent.parent / "img" / "logo.png"


def configure_page() -> None:
    st.set_page_config(
        page_title="Asistente de conocimiento interno",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_styles(), unsafe_allow_html=True)


def render_header() -> None:
    logo = _logo_markup()
    st.markdown(
        f"""
        <section class="topbar">
            <div class="brand">
                <div class="brand-mark">{logo}</div>
                <div>
                    <h1>Asistente de conocimiento interno</h1>
                    <p class="subtitle">
                        Pregunta sobre accesos, RRHH, seguridad o procedimientos internos
                        y recibe una respuesta basada en la documentación corporativa,
                        con fuentes para comprobarla.
                    </p>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _logo_markup() -> str:
    try:
        encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    except OSError:
        return "?"
    return f'<img src="data:image/png;base64,{encoded}" alt="Logo" />'


def render_missing_key() -> None:
    st.error("No se ha encontrado `GOOGLE_API_KEY`.")
    st.info(
        "Crea un archivo `.env` a partir de `.env.template` o exporta la variable "
        "de entorno antes de lanzar la aplicación."
    )


def render_error(message: str) -> None:
    st.error(message)


def render_sidebar_inputs(
    default_top_k: int,
    default_chunk_size: int,
    default_chunk_overlap: int,
    has_active_index: bool,
) -> dict:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-heading">
                <h2>Configuración</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        category_options = category_labels()
        category_label = st.selectbox(
            "Área de soporte",
            category_options,
            index=default_index(),
            help=(
                "Filtra los resultados por área.\n\n"
                "Déjalo en todas las categorías si no quieres limitar la búsqueda."
            ),
        )
        use_query_analysis = st.toggle(
            "Clasificación automática",
            value=True,
            help=(
                "Reformula la solicitud y detecta el área probable.\n\n"
                "No filtra por sí sola. Para limitar resultados, usa el selector de área."
            ),
        )

        with st.expander(SIDEBAR_SECTION_TITLES["technical"], expanded=False):
            chunk_size = st.slider(
                "Tamaño de chunk",
                min_value=200,
                max_value=1500,
                value=default_chunk_size,
                step=50,
                help=(
                    "Tamaño de cada fragmento indexado.\n\n"
                    "Más grande: más contexto.\n\n"
                    "Más pequeño: más precisión."
                ),
            )
            chunk_overlap = st.slider(
                "Solapamiento",
                min_value=0,
                max_value=300,
                value=min(default_chunk_overlap, 300),
                step=10,
                help=(
                    "Texto compartido entre fragmentos.\n\n"
                    "Ayuda cuando una respuesta queda cortada entre chunks."
                ),
            )
            top_k = st.slider(
                "Evidencias recuperadas (k)",
                min_value=1,
                max_value=8,
                value=default_top_k,
                help=(
                    "Número de fragmentos recuperados.\n\n"
                    "Más k: más contexto, pero también más ruido."
                ),
            )

        action_label = "Aplicar ajustes" if has_active_index else "Preparar índice"
        rebuild = st.button(
            action_label,
            type="primary",
            use_container_width=True,
            help="Reconstruye el índice con estos parámetros.",
        )

        st.markdown(
            """
            <div class="provider-footnote">
                Proveedor: Google Gemini · LangChain
            </div>
            """,
            unsafe_allow_html=True,
        )

    category = get_by_label(category_label)
    return {
        "rebuild": rebuild,
        "ingestion": IngestionConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        "retrieval": RetrievalConfig(
            top_k=top_k,
            category_filter=category.name if category is not None else None,
            use_query_analysis=use_query_analysis,
        ),
    }


def render_chat_prompt() -> str | None:
    prompt = st.chat_input("¿Qué necesitas consultar?")
    return prompt or None


def render_index_status(index: IndexHandle | None) -> None:
    if index is None:
        st.markdown(
            """
            <div class="index-status pending">
                <span class="dot"></span>
                <div>
                    <strong>Sin índice activo</strong>
                    <p>Prepara la base de conocimiento para que soporte pueda resolver solicitudes con evidencia.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    distribution = index.categories_distribution
    chips = " ".join(
        _category_chip(category, distribution.get(category.name, 0))
        for category in CATEGORIES
        if distribution.get(category.name, 0) > 0
    )
    st.markdown(
        f"""
        <div class="index-status ready">
                <span class="dot"></span>
                <div>
                <strong>Conocimiento operativo · {escape(index.source_label)}</strong>
                <p>{index.total_chunks} evidencias indexadas · {chips}</p>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )


def render_support_intake(cases: tuple[SuggestedCase, ...]) -> str | None:
    st.markdown(
        """
        <section class="case-intake">
            <div>
                <h2>Preguntas frecuentes</h2>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    selected_question = None
    for row_start in range(0, len(cases), 3):
        row_cases = cases[row_start : row_start + 3]
        columns = st.columns(len(row_cases), gap="small")
        for offset, case in enumerate(row_cases):
            index = row_start + offset
            with columns[offset]:
                st.markdown(
                    f"""
                    <div class="case-card">
                        <div class="case-area">{escape(case.area)}</div>
                        <strong>{escape(case.title)}</strong>
                        <p>{escape(case.impact)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Resolver", key=f"suggested-case-{index}", use_container_width=True):
                    selected_question = case.question
    return selected_question


def render_rag_inspector(
    query: str,
    chunks: tuple[RetrievedChunk, ...],
    analyzed_query: AnalyzedQuery | None = None,
) -> None:
    if analyzed_query is not None:
        category = get_by_name(analyzed_query.category)
        category_display = category.display_name if category else analyzed_query.category
        st.markdown(
            f"""
            <div class="inspector-note">
                <span>Query usada</span>
                <strong>{escape(analyzed_query.query)}</strong>
                <em>{escape(category_display)}</em>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="inspector-note">
                <span>Query usada</span>
                <strong>{escape(query)}</strong>
                <em>Sin query analysis</em>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not chunks:
        st.info("No se ha recuperado ninguna evidencia para esta solicitud.")
        return

    for index, chunk in enumerate(chunks, start=1):
        page = (chunk.page + 1) if isinstance(chunk.page, int) else "?"
        category = get_by_name(chunk.category)
        category_display = category.display_name if category else chunk.category
        with st.expander(
            f"Evidencia {index} · {category_display} · pág. {page} · chunk #{chunk.chunk_id}",
            expanded=index == 1,
        ):
            st.markdown(
                f'<pre class="chunk-preview">{escape(chunk.content)}</pre>',
                unsafe_allow_html=True,
            )


def render_evaluation_results(results: tuple[EvaluationResult, ...]) -> None:
    if not results:
        st.info("Todavía no se ha ejecutado ninguna evaluación.")
        return

    passed = sum(1 for result in results if result.passed)
    retrieval_passed = sum(1 for result in results if result.retrieval_passed)
    answer_passed = sum(1 for result in results if result.answer_passed)
    total = len(results)
    failed = total - passed
    status_label = "Listo para demo" if failed == 0 else "Requiere ajustes"
    status_detail = (
        "Todas las preguntas de control recuperan contexto y generan la respuesta esperada."
        if failed == 0
        else "Hay preguntas de control donde falla la recuperación, la respuesta o el rechazo."
    )
    st.markdown(
        f"""
        <div class="eval-dashboard">
            <div class="eval-status {'ok' if failed == 0 else 'warn'}">
                <span>Estado</span>
                <strong>{escape(status_label)}</strong>
                <p>{escape(status_detail)}</p>
            </div>
            <div class="eval-kpi">
                <span>Pipeline correcto</span>
                <strong>{passed}/{total}</strong>
            </div>
            <div class="eval-kpi">
                <span>Recuperación</span>
                <strong>{retrieval_passed}/{total}</strong>
            </div>
            <div class="eval-kpi">
                <span>Respuesta</span>
                <strong>{answer_passed}/{total}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for result in results:
        state_class = "pass" if result.passed else "fail"
        status = "Correcto" if result.passed else "Revisar"
        st.markdown(
            f"""
            <div class="eval-row {state_class}">
                <div class="eval-case-main">
                    <span class="eval-badge">{escape(status)}</span>
                    <strong>{escape(result.case.title)}</strong>
                    <p>{escape(result.case.question)}</p>
                </div>
                <div class="eval-case-detail">
                    <div>
                        <span>Objetivo</span>
                        <p>{_evaluation_expected_text(result)}</p>
                    </div>
                    <div>
                        <span>Recuperación</span>
                        <p>{_evaluation_retrieval_text(result)}</p>
                    </div>
                    <div>
                        <span>Respuesta</span>
                        <p>{_evaluation_answer_text(result)}</p>
                    </div>
                    <div>
                        <span>Diagnóstico</span>
                        <p>{_evaluation_diagnosis(result)}</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _evaluation_expected_text(result: EvaluationResult) -> str:
    if result.case.should_answer:
        category = _category_display_name(result.case.expected_category)
        return (
            "Responder con documentación de "
            f"{escape(category)}. Señales esperadas: {escape(_format_terms(result.case.expected_terms))}."
        )
    return (
        "No responder si la base documental no contiene el dato solicitado."
    )


def _evaluation_retrieval_text(result: EvaluationResult) -> str:
    evidence_label = "evidencia" if result.retrieved_count == 1 else "evidencias"
    if result.case.expected_category is None:
        category_text = "sin área esperada"
    else:
        category_text = (
            "área esperada encontrada"
            if result.category_hit
            else "área esperada no encontrada"
        )
    if result.matched_terms:
        terms = f"términos encontrados: {escape(_format_terms(result.matched_terms))}"
    else:
        terms = "no aparecen los términos esperados"
    return (
        f"{result.retrieved_count} {evidence_label}; {category_text}; "
        f"{terms}; cobertura {result.term_coverage:.0%}."
    )


def _evaluation_answer_text(result: EvaluationResult) -> str:
    behavior = "rechaza responder" if result.answer_refused else "genera respuesta"
    if result.case.should_answer:
        expected_behavior = "se esperaba respuesta"
        term_text = (
            f"señales en respuesta: {escape(_format_terms(result.answer_matched_terms))}"
            if result.answer_matched_terms
            else "no incluye las señales esperadas en la respuesta"
        )
    else:
        expected_behavior = "se esperaba rechazo"
        term_text = "rechazo correcto" if result.answer_refused else "no rechazó la consulta"
    preview = escape(_shorten(result.answer or "Sin respuesta generada", 140))
    return (
        f"{behavior}; {expected_behavior}; {term_text}; "
        f"cobertura respuesta {result.answer_term_coverage:.0%}. Respuesta: “{preview}”"
    )


def _evaluation_diagnosis(result: EvaluationResult) -> str:
    if result.passed:
        return "El caso completo se comporta como se esperaba."
    if not result.retrieval_passed and not result.answer_passed:
        return "Fallan recuperación y respuesta. Revisa el índice y endurece el rechazo sin contexto."
    if not result.retrieval_passed:
        if not result.case.should_answer:
            return (
                "La recuperación trae señales de un dato que debería quedar sin respuesta. "
                "Revisa chunking, filtros o preguntas de control."
            )
        if result.retrieved_count == 0:
            return "No se recuperó contexto. Revisa el índice, el valor de k o la formulación de la consulta."
        if not result.category_hit:
            return "El filtro o la clasificación están llevando la búsqueda al área equivocada."
        return "El contexto recuperado es débil. Revisa chunking, metadatos o aumenta k."
    if not result.answer_passed and not result.case.should_answer:
        return (
            "El retrieval no encontró una respuesta fiable, pero el modelo contestó. "
            "Hay que reforzar el prompt de rechazo."
        )
    return "La recuperación parece correcta, pero la respuesta no cumple el comportamiento esperado."


def _category_display_name(category_name: str | None) -> str:
    if category_name is None:
        return "ningún área concreta"
    category = get_by_name(category_name)
    return category.display_name if category else category_name


def _format_terms(terms: tuple[str, ...]) -> str:
    return ", ".join(terms) if terms else "sin términos esperados"


def _shorten(value: str, max_chars: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "…"


def render_history(
    turns: list[ChatTurn],
    visible_count: int,
) -> tuple[bool, bool]:
    if not turns:
        return False, False

    clear_clicked = render_conversation_actions(turns)
    ordered_turns = list(reversed(turns))
    visible_turns = ordered_turns[:visible_count]
    has_more = len(ordered_turns) > len(visible_turns)

    for index, turn in enumerate(visible_turns):
        fade_class = _fade_class(index, len(visible_turns))
        st.markdown(_render_turn_html(turn, fade_class), unsafe_allow_html=True)
        _render_turn_expanders(turn, key_suffix=f"history-{turn.created_at.isoformat()}-{index}")

    show_more_clicked = False
    if has_more:
        st.markdown('<div class="show-more-marker">', unsafe_allow_html=True)
        show_more_clicked = st.button(
            "Mostrar más preguntas",
            type="tertiary",
        )
        st.markdown("</div>", unsafe_allow_html=True)
    return clear_clicked, show_more_clicked


def render_streaming_turn(
    placeholder: st.delta_generator.DeltaGenerator,
    question: str,
    partial_answer: str,
    chunks: tuple[RetrievedChunk, ...],
    analyzed_query: AnalyzedQuery | None,
) -> None:
    placeholder.markdown(
        f"""
        <article class="qa-pair streaming-pair">
            {_question_card_html(question, analyzed_query, status_text="Generando…")}
            {_answer_card_html(
                answer_text=partial_answer,
                citations=_chunks_to_citations(chunks),
                status_text="Streaming",
                streaming=True,
                refused=False,
            )}
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


def export_history(turns: list[ChatTurn]) -> None:
    data = _history_to_markdown(turns)
    st.download_button(
        "↓",
        data=data,
        file_name="rag_conversation.md",
        mime="text/markdown",
        help="Exportar conversación",
        disabled=not turns,
        type="tertiary",
    )


# ---------------------------------------------------------------------------
# Internal rendering helpers
# ---------------------------------------------------------------------------


def _render_turn_html(turn: ChatTurn, fade_class: str) -> str:
    refused = turn.status == "no_context"
    return f"""
    <article class="qa-pair {fade_class}">
        {_question_card_html(
            turn.question,
            turn.analyzed_query,
            status_text=turn.created_at.strftime('%Y-%m-%d %H:%M'),
        )}
        {_answer_card_html(
            answer_text=turn.answer,
            citations=turn.citations,
            status_text=_retrieval_status_text(turn),
            streaming=False,
            refused=refused,
        )}
    </article>
    """


def _render_turn_expanders(turn: ChatTurn, key_suffix: str) -> None:
    if not turn.chunks:
        return
    with st.expander("Contexto recuperado", expanded=False):
        for index, chunk in enumerate(turn.chunks, start=1):
            page = (chunk.page + 1) if isinstance(chunk.page, int) else "?"
            category = get_by_name(chunk.category)
            category_display = category.display_name if category else chunk.category
            st.markdown(
                f"**Fragmento {index}** · {escape(chunk.source_name)} · "
                f"pág. {page} · chunk #{chunk.chunk_id} · {escape(category_display)}"
            )
            st.markdown(
                f'<pre class="chunk-preview">{escape(chunk.content)}</pre>',
                unsafe_allow_html=True,
            )


def _question_card_html(
    question: str,
    analyzed_query: AnalyzedQuery | None,
    status_text: str,
) -> str:
    analyzed_block = ""
    if analyzed_query is not None:
        category = get_by_name(analyzed_query.category)
        category_display = category.display_name if category else analyzed_query.category
        accent = escape(category.accent if category else "#5b625f")
        analyzed_block = (
            '<div class="rewrite-row">'
            "<span>Consulta reescrita</span>"
            f"<code>{escape(analyzed_query.query)}</code>"
            f'<span class="category-pill" style="--accent: {accent}">'
            f"{escape(category_display)}</span>"
            "</div>"
        )
    return (
        '<section class="message-card source-card">'
        '<div class="message-meta">'
        "<span>Pregunta</span>"
        f"<span>{escape(status_text)}</span>"
        "</div>"
        f'<div class="message-body">{_format_text(question)}</div>'
        f"{analyzed_block}"
        "</section>"
    )


def _answer_card_html(
    answer_text: str,
    citations: tuple[Citation, ...],
    status_text: str,
    streaming: bool,
    refused: bool,
) -> str:
    if streaming and not answer_text:
        body_html = '<span class="stream-cursor">▌</span>'
    else:
        body_html = _format_answer_markdown(answer_text)

    citations_html = ""
    if citations:
        chips = "".join(_citation_chip_html(citation) for citation in citations)
        citations_html = f'<div class="citation-row">{chips}</div>'

    state_class = "refused" if refused else ""
    return (
        f'<section class="message-card target-card {state_class}">'
        '<div class="message-meta">'
        "<span>Respuesta</span>"
        f"<span>{escape(status_text)}</span>"
        "</div>"
        f'<div class="message-body">{body_html}</div>'
        f"{citations_html}"
        "</section>"
    )


def _citation_chip_html(citation: Citation) -> str:
    page = (citation.page + 1) if isinstance(citation.page, int) else "?"
    category = get_by_name(citation.category)
    category_display = category.display_name if category else citation.category
    accent = category.accent if category else "#f5660b"
    return (
        f'<span class="citation-chip" style="--accent: {escape(accent)}">'
        f'<strong>{escape(citation.source_name)}</strong> · pág. {page} · '
        f'chunk #{citation.chunk_id} · {escape(category_display)}'
        f"</span>"
    )


def _section_label(text: str) -> None:
    st.markdown(
        f'<div class="sidebar-section">{escape(text)}</div>',
        unsafe_allow_html=True,
    )


def _category_chip(category: Category, count: int) -> str:
    return (
        f'<span class="category-pill compact" style="--accent: {escape(category.accent)}">'
        f"{escape(category.display_name)} · {count}"
        f"</span>"
    )


def _retrieval_status_text(turn: ChatTurn) -> str:
    if turn.status == "no_context":
        return "Sin respuesta · sin contexto suficiente"
    parts = [f"{len(turn.chunks)} fragmentos · k={turn.retrieval.top_k}"]
    if turn.retrieval.use_query_analysis:
        parts.append("query-analysis")
    if turn.retrieval.category_filter:
        category = get_by_name(turn.retrieval.category_filter)
        if category is not None:
            parts.append(f"filtro: {category.display_name}")
    return " · ".join(parts)


def _chunks_to_citations(chunks: tuple[RetrievedChunk, ...]) -> tuple[Citation, ...]:
    return tuple(
        Citation(
            source_name=chunk.source_name,
            page=chunk.page,
            chunk_id=chunk.chunk_id,
            category=chunk.category,
        )
        for chunk in chunks
    )


def _history_to_markdown(turns: list[ChatTurn]) -> str:
    if not turns:
        return ""

    chunks_md = ["# Mesa de ayuda interna - conversación RAG", ""]
    for index, turn in enumerate(turns, start=1):
        chunks_md.extend(
            [
                f"## Turn {index}",
                "",
                f"- Created at: {turn.created_at.isoformat(timespec='seconds')}",
                f"- Status: {turn.status}",
                f"- Top-K: {turn.retrieval.top_k}",
                f"- Category filter: {turn.retrieval.category_filter or 'all'}",
                f"- Query analysis: {turn.retrieval.use_query_analysis}",
                "",
                "### Question",
                "",
                turn.question,
                "",
                "### Effective query",
                "",
                turn.effective_query,
                "",
                "### Answer",
                "",
                turn.answer,
                "",
            ]
        )
        if turn.citations:
            chunks_md.append("### Sources")
            chunks_md.append("")
            for citation in turn.citations:
                page = (citation.page + 1) if isinstance(citation.page, int) else "?"
                chunks_md.append(
                    f"- {citation.source_name} · page {page} · chunk #{citation.chunk_id} · {citation.category}"
                )
            chunks_md.append("")
    return "\n".join(chunks_md)


def _format_text(value: str) -> str:
    if not value:
        return ""
    return "<br>".join(escape(value).splitlines())


def _format_answer_markdown(value: str) -> str:
    if not value:
        return ""

    html: list[str] = []
    in_ordered_list = False
    in_unordered_list = False

    def close_lists() -> None:
        nonlocal in_ordered_list, in_unordered_list
        if in_ordered_list:
            html.append("</ol>")
            in_ordered_list = False
        if in_unordered_list:
            html.append("</ul>")
            in_unordered_list = False

    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            close_lists()
            continue

        heading = re.match(r"^#{1,4}\s+(.+)$", line)
        ordered_item = re.match(r"^\d+\.\s+(.+)$", line)
        unordered_item = re.match(r"^[-*]\s+(.+)$", line)

        if heading:
            close_lists()
            html.append(f"<h3>{_format_inline_markdown(heading.group(1))}</h3>")
        elif ordered_item:
            if in_unordered_list:
                html.append("</ul>")
                in_unordered_list = False
            if not in_ordered_list:
                html.append("<ol>")
                in_ordered_list = True
            html.append(f"<li>{_format_inline_markdown(ordered_item.group(1))}</li>")
        elif unordered_item:
            if in_ordered_list:
                html.append("</ol>")
                in_ordered_list = False
            if not in_unordered_list:
                html.append("<ul>")
                in_unordered_list = True
            html.append(f"<li>{_format_inline_markdown(unordered_item.group(1))}</li>")
        else:
            close_lists()
            html.append(f"<p>{_format_inline_markdown(line)}</p>")

    close_lists()
    return f'<div class="answer-markdown">{"".join(html)}</div>'


def _format_inline_markdown(value: str) -> str:
    formatted = escape(value)
    formatted = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", formatted)
    formatted = re.sub(r"`([^`]+)`", r"<code>\1</code>", formatted)
    return formatted


def _fade_class(index: int, total: int) -> str:
    if total < 4:
        return ""
    if index == total - 1:
        return "fade-strong"
    if index == total - 2:
        return "fade-soft"
    return ""


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

        html,
        body {
            min-height: 100%;
            overflow-y: auto;
        }

        [data-testid="stAppViewContainer"] {
            min-height: 100vh;
            overflow: visible;
        }

        [data-testid="stMain"] {
            min-height: 100vh;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            scrollbar-gutter: stable;
        }

        .block-container {
            max-width: 1180px;
            min-height: 100vh;
            padding-top: 1.25rem;
            padding-bottom: 5.4rem;
            overflow: visible;
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
            gap: 18px;
            margin-bottom: 10px;
            padding: 22px 24px;
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

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-pair),
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.empty-state) {
            height: clamp(320px, 48vh, 560px) !important;
            min-height: 320px;
            padding: 8px 12px 82px 4px;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            overscroll-behavior: contain;
            scrollbar-gutter: stable;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-pair) > div,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.empty-state) > div {
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
            width: 64px;
            height: 64px;
            flex: 0 0 64px;
            place-items: center;
            overflow: hidden;
            color: var(--green);
            background: #ffffff;
            border: 1px solid rgba(20, 63, 53, 0.14);
            border-radius: 50%;
            box-shadow: 0 10px 26px rgba(20, 63, 53, 0.12);
            font-size: 1.55rem;
            font-weight: 900;
        }

        .brand-mark img {
            display: block;
            width: 100%;
            height: 100%;
            object-fit: contain;
            transform: scale(1.65);
        }

        .topbar h1 {
            color: var(--ink);
            font-size: 1.72rem;
            line-height: 1.15;
            margin: 0.05rem 0 0.35rem;
            letter-spacing: 0;
        }

        .subtitle {
            color: var(--muted);
            font-size: 0.95rem;
            margin: 0;
            max-width: 720px;
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

        .sidebar-section {
            margin: 20px 0 6px;
            color: rgba(255, 255, 255, 0.62);
            font-size: 0.7rem;
            font-weight: 900;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }

        [data-testid="stSidebar"] div[data-testid="stSelectbox"],
        [data-testid="stSidebar"] div[data-testid="stSlider"],
        [data-testid="stSidebar"] div[data-testid="stToggle"] {
            margin-bottom: 10px;
        }

        [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] span,
        [data-testid="stSidebar"] [data-testid="stToggle"] p,
        [data-testid="stSidebar"] [data-testid="stToggle"] span,
        [data-testid="stSidebar"] [data-testid="stToggle"] div {
            color: #ffffff !important;
            font-weight: 700;
        }

        [data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg.icon {
            stroke: #ffffff !important;
        }

        [data-testid="stSidebar"] [data-testid="stTooltipIcon"] button {
            color: #ffffff !important;
        }

        div[data-baseweb="popover"],
        div[role="tooltip"] {
            max-width: 300px !important;
            z-index: 999999 !important;
        }

        div[data-baseweb="popover"]:has([data-testid="stTooltipContent"]),
        div[role="tooltip"]:has([data-testid="stMarkdownContainer"]) {
            width: 300px !important;
            max-width: min(300px, calc(100vw - 32px)) !important;
        }

        [data-testid="stTooltipContent"],
        div[role="tooltip"] [data-testid="stMarkdownContainer"] {
            box-sizing: border-box !important;
            width: 280px !important;
            max-width: min(280px, calc(100vw - 32px)) !important;
            max-height: 170px !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            white-space: normal !important;
            overflow-wrap: anywhere !important;
            font-size: 0.82rem !important;
            line-height: 1.35 !important;
        }

        [data-testid="stTooltipContent"] *,
        div[role="tooltip"] [data-testid="stMarkdownContainer"] * {
            font-size: 0.82rem !important;
            line-height: 1.35 !important;
            white-space: normal !important;
            overflow-wrap: anywhere !important;
        }

        [data-testid="stTooltipContent"] p,
        div[role="tooltip"] [data-testid="stMarkdownContainer"] p {
            margin: 0 0 0.45rem !important;
            font-size: 0.82rem !important;
            line-height: 1.35 !important;
            white-space: normal !important;
            overflow-wrap: anywhere !important;
        }

        [data-testid="stTooltipContent"] p:last-child,
        div[role="tooltip"] [data-testid="stMarkdownContainer"] p:last-child {
            margin-bottom: 0 !important;
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

        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {
            background: var(--orange) !important;
            border-color: var(--orange) !important;
            color: #ffffff !important;
            font-weight: 800 !important;
            border-radius: 8px !important;
            box-shadow: 0 12px 26px rgba(245, 102, 11, 0.32) !important;
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 32px rgba(245, 102, 11, 0.4) !important;
        }

        [data-testid="stSidebar"] details {
            margin: 18px 0 14px;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 8px;
        }

        [data-testid="stSidebar"] details:hover {
            background: rgba(255, 255, 255, 0.11);
            border-color: rgba(255, 255, 255, 0.28);
        }

        [data-testid="stSidebar"] details summary {
            min-height: 48px;
            padding: 10px 14px !important;
            color: #ffffff !important;
            background: rgba(255, 255, 255, 0.08);
            font-weight: 800;
        }

        [data-testid="stSidebar"] details summary * {
            color: #ffffff !important;
            font-weight: 800 !important;
        }

        [data-testid="stSidebar"] details[open] summary {
            background: rgba(255, 255, 255, 0.16);
            border-bottom: 1px solid rgba(255, 255, 255, 0.12);
        }

        [data-testid="stSidebar"] details > div {
            padding: 14px 14px 16px;
        }

        [data-testid="stSidebar"] details [data-testid="stSlider"] {
            margin: 0 0 24px;
        }

        [data-testid="stSidebar"] details [data-testid="stSlider"] [data-testid="stWidgetLabel"] p {
            color: #ffffff !important;
            font-size: 0.86rem;
            font-weight: 800;
        }

        [data-testid="stSidebar"] details [data-baseweb="slider"] {
            margin-top: 8px;
            padding: 18px 0 22px;
            min-height: 58px;
        }

        [data-testid="stSidebar"] details [data-baseweb="slider"] div[role="slider"] {
            background: #ffffff !important;
            border: 2px solid var(--orange) !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.22) !important;
        }

        [data-testid="stSidebar"] details [data-baseweb="slider"] .e2ups022 {
            top: -1.55rem !important;
            padding: 0 !important;
            color: rgba(255, 255, 255, 0.9) !important;
            background: transparent !important;
            font-size: 0.82rem !important;
            font-weight: 800 !important;
            line-height: 1 !important;
        }

        [data-testid="stSidebar"] details [data-testid="stSliderTickBar"] {
            opacity: 1 !important;
            top: 42px !important;
            margin-top: 0 !important;
            padding: 0 2px !important;
            color: rgba(255, 255, 255, 0.76) !important;
            transition: none !important;
        }

        [data-testid="stSidebar"] details [data-testid="stSliderTickBar"] *,
        [data-testid="stSidebar"] details [data-testid="stSliderTickBar"] div,
        [data-testid="stSidebar"] details [data-testid="stSliderTickBar"] p,
        [data-testid="stSidebar"] details [data-testid="stSliderTickBar"] span {
            color: rgba(255, 255, 255, 0.76) !important;
            background: transparent !important;
            box-shadow: none !important;
            font-weight: 700 !important;
            opacity: 1 !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] {
            margin: 18px 0 14px;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.08) !important;
            border: 1px solid rgba(255, 255, 255, 0.16) !important;
            border-radius: 8px !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] details {
            margin: 0;
            background: transparent;
            border: 0;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary * {
            color: #ffffff !important;
            font-weight: 800 !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
            padding: 14px !important;
            background: rgba(13, 24, 40, 0.18) !important;
        }

        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.16);
            margin: 26px 0 18px;
        }

        .provider-footnote {
            margin-top: 56px;
            padding: 0 2px 8px;
            color: rgba(255, 255, 255, 0.42);
            font-size: 0.68rem;
            line-height: 1.3;
            letter-spacing: 0;
        }

        .index-status {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            margin: 0 0 14px;
            padding: 14px 16px;
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 12px 26px rgba(13, 24, 40, 0.06);
        }

        .index-status .dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            margin-top: 6px;
            border-radius: 50%;
            background: var(--green);
            box-shadow: 0 0 0 4px rgba(20, 63, 53, 0.16);
        }

        .index-status.pending .dot {
            background: var(--orange);
            box-shadow: 0 0 0 4px rgba(245, 102, 11, 0.18);
        }

        .index-status strong {
            display: block;
            font-size: 0.95rem;
            color: var(--ink);
            margin-bottom: 4px;
        }

        .index-status p {
            margin: 0;
            color: var(--muted);
            font-size: 0.85rem;
            line-height: 1.45;
        }

        .case-intake {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 18px;
            margin: 2px 0 12px;
        }

        .case-intake h2 {
            margin: 2px 0 0;
            color: var(--ink);
            font-size: 1.12rem;
            line-height: 1.2;
        }

        .case-intake p {
            max-width: 500px;
            margin: 0;
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.45;
        }

        .eyebrow {
            color: var(--orange);
            font-size: 0.7rem;
            font-weight: 900;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .case-card {
            min-height: 124px;
            padding: 13px 14px;
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 10px 24px rgba(13, 24, 40, 0.05);
        }

        .case-card strong {
            display: block;
            color: var(--ink);
            margin: 6px 0;
            font-size: 0.98rem;
            line-height: 1.25;
        }

        .case-card p {
            margin: 0;
            color: var(--muted);
            font-size: 0.78rem;
            line-height: 1.38;
        }

        .case-area {
            color: var(--green);
            font-size: 0.68rem;
            font-weight: 900;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }

        .inspector-note {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 2px 0 12px;
            padding: 10px 12px;
            background: var(--navy-soft);
            border: 1px solid var(--line);
            border-radius: 8px;
            color: var(--muted);
            font-size: 0.78rem;
        }

        .inspector-note span {
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .inspector-note strong {
            color: var(--ink);
            font-size: 0.9rem;
            overflow-wrap: anywhere;
        }

        .inspector-note em {
            margin-left: auto;
            color: var(--green);
            font-style: normal;
            font-weight: 800;
            white-space: nowrap;
        }

        .eval-dashboard {
            display: grid;
            grid-template-columns: minmax(280px, 1.7fr) repeat(3, minmax(150px, 1fr));
            gap: 10px;
            margin: 8px 0 14px;
        }

        .eval-status,
        .eval-kpi {
            padding: 14px 16px;
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 10px 24px rgba(13, 24, 40, 0.05);
        }

        .eval-status {
            border-left: 4px solid var(--orange);
        }

        .eval-status.ok {
            border-left-color: var(--green);
        }

        .eval-status span,
        .eval-kpi span,
        .eval-case-detail span {
            display: block;
            margin-bottom: 5px;
            color: var(--muted);
            font-size: 0.68rem;
            font-weight: 900;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }

        .eval-status strong,
        .eval-kpi strong {
            display: block;
            color: var(--ink);
            font-size: 1.15rem;
            line-height: 1.2;
        }

        .eval-status p {
            margin: 6px 0 0;
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.35;
        }

        .eval-row {
            display: grid;
            grid-template-columns: minmax(260px, 0.9fr) minmax(0, 1.6fr);
            gap: 18px;
            margin-bottom: 10px;
            padding: 16px;
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid var(--line);
            border-left: 4px solid var(--orange);
            border-radius: 8px;
        }

        .eval-row.pass {
            border-left-color: var(--green);
        }

        .eval-case-main strong {
            display: block;
            color: var(--ink);
            margin: 8px 0 4px;
            font-size: 1rem;
        }

        .eval-case-main p,
        .eval-case-detail p {
            margin: 0;
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.42;
        }

        .eval-badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 8px;
            color: #ffffff;
            background: var(--orange);
            border-radius: 999px;
            font-size: 0.68rem;
            font-weight: 900;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .eval-row.pass .eval-badge {
            background: var(--green);
        }

        .eval-case-detail {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
        }

        .eval-case-detail > div {
            padding: 11px 12px;
            background: var(--navy-soft);
            border-radius: 8px;
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

        .qa-pair {
            display: grid;
            grid-template-columns: minmax(0, 1fr);
            gap: 10px;
            margin: 0 0 22px;
        }

        .qa-pair.fade-soft {
            opacity: 0.58;
            filter: saturate(0.72);
        }

        .qa-pair.fade-strong {
            opacity: 0.32;
            filter: saturate(0.7);
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
            padding: 15px 17px;
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

        .target-card.refused {
            background: linear-gradient(180deg, rgba(13, 24, 40, 0.96) 0%, rgba(20, 63, 53, 0.96) 100%);
            border-color: rgba(245, 102, 11, 0.6);
            box-shadow: 0 14px 32px rgba(13, 24, 40, 0.18);
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
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }

        .target-card .message-meta {
            color: rgba(255, 255, 255, 0.7);
        }

        .message-body {
            color: inherit;
            font-size: 1rem;
            line-height: 1.55;
            overflow-wrap: anywhere;
        }

        .answer-markdown h3 {
            margin: 1.1rem 0 0.45rem;
            color: #ffffff;
            font-size: 1.05rem;
            line-height: 1.25;
        }

        .answer-markdown h3:first-child {
            margin-top: 0;
        }

        .answer-markdown p {
            margin: 0 0 0.75rem;
        }

        .answer-markdown ol,
        .answer-markdown ul {
            margin: 0 0 0.9rem 1.25rem;
            padding: 0;
        }

        .answer-markdown li {
            margin: 0.18rem 0;
            padding-left: 0.2rem;
        }

        .answer-markdown code {
            padding: 2px 5px;
            color: #ffffff;
            background: rgba(255, 255, 255, 0.12);
            border-radius: 5px;
        }

        .source-card .message-body {
            color: var(--ink);
        }

        .rewrite-row {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px dashed rgba(13, 24, 40, 0.18);
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .rewrite-row code {
            color: var(--navy);
            background: rgba(255, 255, 255, 0.6);
            border: 1px solid rgba(13, 24, 40, 0.08);
            border-radius: 6px;
            padding: 4px 8px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: none;
        }

        .citation-row {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 14px;
            padding-top: 12px;
            border-top: 1px solid rgba(255, 255, 255, 0.18);
        }

        .citation-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-left: 3px solid var(--accent, var(--orange));
            border-radius: 6px;
            color: rgba(255, 255, 255, 0.92);
            font-size: 0.72rem;
            font-weight: 600;
        }

        .citation-chip strong {
            color: #ffffff;
            font-weight: 800;
        }

        .category-pill {
            display: inline-flex;
            align-items: center;
            padding: 3px 8px;
            background: rgba(13, 24, 40, 0.06);
            border-radius: 999px;
            color: var(--accent, var(--navy));
            font-size: 0.7rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .category-pill.compact {
            background: rgba(13, 24, 40, 0.05);
            margin-right: 6px;
            font-size: 0.68rem;
        }

        .chunk-preview {
            margin: 4px 0 14px;
            padding: 12px 14px;
            background: var(--navy-soft);
            border: 1px solid var(--line);
            border-radius: 8px;
            color: var(--ink);
            font-size: 0.88rem;
            line-height: 1.5;
            white-space: pre-wrap;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
        }

        .empty-state {
            display: grid;
            place-items: center;
            min-height: 280px;
            background: rgba(255, 255, 255, 0.96);
            border: 1px dashed #b9d7cd;
            border-radius: 8px;
            margin-top: 0.6rem;
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
            max-width: 480px;
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
                flex-direction: column;
                align-items: flex-start;
            }

            .topbar h1 {
                font-size: 1.55rem;
            }

            .provider-footnote {
                position: static;
                width: auto;
                margin-top: 28px;
            }

            .citation-row {
                gap: 4px;
            }

            .case-intake {
                align-items: flex-start;
                flex-direction: column;
            }

            .eval-dashboard {
                grid-template-columns: minmax(0, 1fr);
            }

            .eval-row {
                grid-template-columns: minmax(0, 1fr);
            }

            .eval-case-detail {
                grid-template-columns: minmax(0, 1fr);
            }
        }
    </style>
    """
