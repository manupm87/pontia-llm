from __future__ import annotations

import streamlit as st

from core.config import Settings, load_settings
from core.models import (
    AskRequest,
    ChatTurn,
    IngestionConfig,
)
from core.rag_engine import (
    GenerationError,
    IndexBuildError,
    IndexHandle,
    RagEngine,
    REFUSAL_SENTENCE,
    citations_from_chunks,
)
from core.state import (
    add_turn,
    autobuild_attempted,
    clear_history,
    history,
    index_handle,
    init_state,
    mark_autobuild_attempted,
    reset_autobuild_flag,
    set_index_handle,
    show_more_history,
    visible_history_count,
)
from core.ui import (
    configure_page,
    render_chat_prompt,
    render_evaluation_results,
    render_error,
    render_header,
    render_history,
    render_missing_key,
    render_rag_inspector,
    render_sidebar_inputs,
    render_support_intake,
    render_streaming_turn,
)
from core.use_cases import (
    EVALUATION_CASES,
    SUGGESTED_CASES,
    evaluate_retrieved_chunks,
)


ENGINE_CACHE_VERSION = 1
CONVERSATION_PANEL_HEIGHT = 520


@st.cache_resource(show_spinner=False)
def get_engine(
    generation_model: str,
    embedding_model: str,
    timeout: float,
    cache_version: int,
) -> RagEngine:
    return RagEngine(
        generation_model=generation_model,
        embedding_model=embedding_model,
        timeout=timeout,
    )


def main() -> None:
    configure_page()
    init_state()

    settings = load_settings()
    render_header()

    if not settings.has_api_key:
        render_missing_key()
        return

    inputs = render_sidebar_inputs(
        default_top_k=settings.default_top_k,
        default_chunk_size=settings.default_chunk_size,
        default_chunk_overlap=settings.default_chunk_overlap,
        has_active_index=index_handle() is not None,
    )

    engine = get_engine(
        generation_model=settings.generation_model,
        embedding_model=settings.embedding_model,
        timeout=settings.request_timeout,
        cache_version=ENGINE_CACHE_VERSION,
    )

    if inputs["rebuild"]:
        _rebuild_index(engine, settings, inputs["ingestion"])
        st.rerun()

    _autobuild_default_index_if_needed(engine, settings, inputs["ingestion"])

    active_index = index_handle()

    if active_index is None:
        prompt = render_chat_prompt()
        if prompt:
            st.warning("Construye primero la base de conocimiento desde la barra lateral.")
        return

    solve_tab, evidence_tab, eval_tab = st.tabs(
        ["Resolver solicitudes", "Auditar recuperación", "Evaluar RAG"]
    )

    with solve_tab:
        selected_question = render_support_intake(SUGGESTED_CASES)
        prompt = selected_question or render_chat_prompt()
        request = (
            AskRequest(question=prompt, retrieval=inputs["retrieval"])
            if prompt is not None
            else None
        )
        _render_conversation(active_index, streaming_request=request, engine=engine)

    with evidence_tab:
        _render_retrieval_audit(engine, active_index, inputs["retrieval"])

    with eval_tab:
        _render_rag_evaluation(engine, active_index, inputs["retrieval"])


def _render_conversation(
    active_index: IndexHandle,
    streaming_request: AskRequest | None,
    engine: RagEngine,
) -> None:
    with st.container(height=CONVERSATION_PANEL_HEIGHT, border=False):
        placeholder = st.empty() if streaming_request else None
        clear_clicked, show_more_clicked = render_history(history(), visible_history_count())

        if streaming_request is not None and placeholder is not None:
            try:
                turn = _answer_streaming(engine, active_index, streaming_request, placeholder)
            except (GenerationError, IndexBuildError) as exc:
                render_error(str(exc))
                return
            add_turn(turn)
            st.rerun()

    if clear_clicked:
        clear_history()
        st.rerun()
    if show_more_clicked:
        show_more_history()
        st.rerun()


def _render_retrieval_audit(
    engine: RagEngine,
    active_index: IndexHandle,
    retrieval,
) -> None:
    st.markdown("#### Inspección de evidencias")
    query = st.text_input(
        "Solicitud a inspeccionar",
        value="No recuerdo mi clave del email de la empresa. ¿Qué hago?",
        help=(
            "Escribe una query.\n\n"
            "Ejecuta la inspección para ver los chunks recuperados."
        ),
    )
    use_analysis = st.toggle(
        "Usar análisis de la query en esta inspección",
        value=retrieval.use_query_analysis,
    )
    inspect_clicked = st.button("Inspeccionar evidencias", type="primary")
    if not inspect_clicked:
        return
    if not query:
        return

    analyzed_query = None
    effective_query = query
    try:
        if use_analysis:
            analyzed_query = engine.analyze_query(query)
            effective_query = analyzed_query.query
        chunks = tuple(
            engine.retrieve(
                index=active_index,
                query=effective_query,
                retrieval=retrieval,
            )
        )
    except (GenerationError, IndexBuildError) as exc:
        render_error(str(exc))
        return

    render_rag_inspector(effective_query, chunks, analyzed_query)


def _render_rag_evaluation(
    engine: RagEngine,
    active_index: IndexHandle,
    retrieval,
) -> None:
    st.markdown("#### Evaluación del RAG")
    st.caption(
        "Ejecuta preguntas de control contra el pipeline completo: recuperación, generación y rechazo cuando no hay contexto suficiente."
    )
    run_eval = st.button("Ejecutar evaluación", type="primary")
    if not run_eval:
        render_evaluation_results(())
        return

    results = []
    progress = st.progress(0, text="Preparando evaluación…")
    for index, case in enumerate(EVALUATION_CASES, start=1):
        progress.progress(
            (index - 1) / len(EVALUATION_CASES),
            text=f"Evaluando: {case.title}",
        )
        analyzed_query = None
        effective_query = case.question
        try:
            if retrieval.use_query_analysis:
                analyzed_query = engine.analyze_query(case.question)
                effective_query = analyzed_query.query
            chunks = tuple(
                engine.retrieve(
                    index=active_index,
                    query=effective_query,
                    retrieval=retrieval,
                )
            )
            answer = "".join(engine.stream_answer(case.question, chunks)).strip()
        except (GenerationError, IndexBuildError) as exc:
            progress.empty()
            render_error(str(exc))
            return
        results.append(evaluate_retrieved_chunks(case, chunks, answer, REFUSAL_SENTENCE))

    progress.progress(1.0, text="Evaluación completada.")
    render_evaluation_results(tuple(results))


def _answer_streaming(
    engine: RagEngine,
    active_index: IndexHandle,
    request: AskRequest,
    placeholder: st.delta_generator.DeltaGenerator,
) -> ChatTurn:
    analyzed_query = None
    effective_query = request.question

    if request.retrieval.use_query_analysis:
        analyzed_query = engine.analyze_query(request.question)
        effective_query = analyzed_query.query

    retrieved = engine.retrieve(
        index=active_index,
        query=effective_query,
        retrieval=request.retrieval,
    )
    chunks_tuple = tuple(retrieved)

    render_streaming_turn(
        placeholder=placeholder,
        question=request.question,
        partial_answer="",
        chunks=chunks_tuple,
        analyzed_query=analyzed_query,
    )

    partial = ""
    for delta in engine.stream_answer(request.question, chunks_tuple):
        partial += delta
        render_streaming_turn(
            placeholder=placeholder,
            question=request.question,
            partial_answer=partial,
            chunks=chunks_tuple,
            analyzed_query=analyzed_query,
        )

    status = "no_context" if (not chunks_tuple or _is_refusal(partial)) else "answered"
    citations = citations_from_chunks(chunks_tuple) if status == "answered" else ()

    return ChatTurn(
        question=request.question,
        answer=partial.strip(),
        chunks=chunks_tuple,
        citations=citations,
        analyzed_query=analyzed_query,
        effective_query=effective_query,
        retrieval=request.retrieval,
        status=status,
    )


def _rebuild_index(
    engine: RagEngine,
    settings: Settings,
    ingestion: IngestionConfig,
) -> None:
    try:
        with st.spinner(f"Indexando {settings.default_pdf_path.name}…"):
            handle = engine.build_index_from_path(
                pdf_path=settings.default_pdf_path,
                ingestion=ingestion,
            )
    except IndexBuildError as exc:
        render_error(str(exc))
        return

    set_index_handle(handle, handle.fingerprint, handle.source_label)
    reset_autobuild_flag()
    clear_history()
    st.success(
        f"Índice listo · {handle.total_chunks} fragmentos extraídos de {handle.source_label}.",
        icon="✅",
    )


def _autobuild_default_index_if_needed(
    engine: RagEngine,
    settings: Settings,
    ingestion: IngestionConfig,
) -> None:
    if index_handle() is not None:
        return
    if autobuild_attempted():
        return
    if not settings.default_pdf_path.exists():
        mark_autobuild_attempted()
        return
    try:
        with st.spinner("Indexando base de conocimiento por defecto…"):
            handle = engine.build_index_from_path(
                pdf_path=settings.default_pdf_path,
                ingestion=ingestion,
            )
    except IndexBuildError:
        mark_autobuild_attempted()
        return
    set_index_handle(handle, handle.fingerprint, handle.source_label)
    mark_autobuild_attempted()


def _is_refusal(text: str) -> bool:
    return REFUSAL_SENTENCE.casefold() in text.casefold()


if __name__ == "__main__":
    main()
