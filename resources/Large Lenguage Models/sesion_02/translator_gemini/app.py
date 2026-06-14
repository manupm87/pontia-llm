from __future__ import annotations

from dataclasses import replace

import streamlit as st

from core.config import load_settings
from core.languages import get_by_name
from core.models import PendingLanguageMismatch, TranslationRequest, TranslationResult
from core.gemini_translator import GeminiTranslator, TranslationError
from core.state import (
    add_turn,
    add_unavailable_turn,
    clear_history,
    clear_pending_mismatch,
    history,
    init_state,
    pending_mismatch,
    request_with_detected_source,
    set_pending_mismatch,
    show_more_history,
    visible_history_count,
)
from core.ui import (
    configure_page,
    render_error,
    render_header,
    render_history,
    render_language_mismatch,
    render_missing_key,
    render_sidebar,
    render_streaming_turn,
)

TRANSLATOR_CACHE_VERSION = 1
CONVERSATION_PANEL_HEIGHT = 560


@st.cache_resource(show_spinner=False)
def get_translator(
    api_key: str,
    model: str,
    timeout: float,
    cache_version: int,
) -> GeminiTranslator:
    return GeminiTranslator(api_key=api_key, model=model, timeout=timeout)


def main() -> None:
    configure_page()
    init_state()

    settings = load_settings()
    render_header()

    if not settings.has_api_key:
        render_missing_key()
        return

    request = render_sidebar(settings.model)
    translator = get_translator(
        api_key=settings.google_api_key or "",
        model=settings.model,
        timeout=settings.request_timeout,
        cache_version=TRANSLATOR_CACHE_VERSION,
    )

    pending = pending_mismatch()
    if pending is not None:
        with st.container(height=CONVERSATION_PANEL_HEIGHT, border=False):
            action = render_language_mismatch(pending)
            clear_clicked, show_more_clicked = render_history(
                history(),
                visible_history_count(),
            )
        if clear_clicked:
            clear_history()
            clear_pending_mismatch()
            st.rerun()
        if show_more_clicked:
            show_more_history()
            st.rerun()
        if action == "accept":
            accepted_request = request_with_detected_source(pending)
            with st.container(height=CONVERSATION_PANEL_HEIGHT, border=False):
                placeholder = st.empty()
                render_history(history(), visible_history_count())
                try:
                    result = _stream_translation(translator, accepted_request, placeholder)
                except TranslationError as exc:
                    render_error(str(exc))
                    return
            add_turn(accepted_request, result)
            clear_pending_mismatch()
            st.rerun()
        if action == "reject":
            add_unavailable_turn(
                pending.request,
                pending.detected_source_language_display,
            )
            clear_pending_mismatch()
            st.rerun()
        return

    if request is None:
        with st.container(height=CONVERSATION_PANEL_HEIGHT, border=False):
            clear_clicked, show_more_clicked = render_history(
                history(),
                visible_history_count(),
            )
        if clear_clicked:
            clear_history()
            st.rerun()
        if show_more_clicked:
            show_more_history()
            st.rerun()
        return

    try:
        final_request = _prepare_request_for_stream(translator, request)
        pending = _preflight_mismatch_if_needed(translator, final_request)
    except TranslationError as exc:
        render_error(str(exc))
        return

    if pending is not None:
        set_pending_mismatch(pending)
        st.rerun()

    with st.container(height=CONVERSATION_PANEL_HEIGHT, border=False):
        placeholder = st.empty()
        render_history(history(), visible_history_count())
        try:
            result = _stream_translation(translator, final_request, placeholder)
        except TranslationError as exc:
            render_error(str(exc))
            return

    add_turn(final_request, result)
    st.rerun()


def _prepare_request_for_stream(
    translator: GeminiTranslator,
    request: TranslationRequest,
) -> TranslationRequest:
    if request.source_language != "Auto-detect":
        return request

    detected_language = get_by_name(translator.detect_language(request.text))
    if detected_language is None:
        return request

    return replace(
        request,
        source_language=detected_language.name,
        source_language_display=detected_language.display_name,
    )


def _preflight_mismatch_if_needed(
    translator: GeminiTranslator,
    request: TranslationRequest,
) -> PendingLanguageMismatch | None:
    if request.source_language == "Auto-detect":
        return None

    detected_language = get_by_name(translator.detect_language(request.text))
    if detected_language is None:
        return None

    if detected_language.name == request.source_language:
        return None

    return PendingLanguageMismatch(
        request=request,
        detected_source_language=detected_language.name,
        detected_source_language_display=detected_language.display_name,
    )


def _stream_translation(
    translator: GeminiTranslator,
    request: TranslationRequest,
    placeholder: st.delta_generator.DeltaGenerator,
) -> TranslationResult:
    translation = ""
    render_streaming_turn(placeholder, request, translation)

    for delta in translator.stream_translation(request):
        translation += delta
        render_streaming_turn(placeholder, request, translation)

    return TranslationResult(
        translation=translation,
        detected_source_language=request.source_language,
        source_language=request.source_language,
        target_language=request.target_language,
        notes=[],
    )


if __name__ == "__main__":
    main()
