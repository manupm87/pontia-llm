from __future__ import annotations

from datetime import datetime
from dataclasses import replace

import streamlit as st

from core.models import ChatTurn, PendingLanguageMismatch, TranslationRequest, TranslationResult


HISTORY_KEY = "translation_history"
PENDING_MISMATCH_KEY = "pending_language_mismatch"
VISIBLE_HISTORY_KEY = "visible_translation_count"
DEFAULT_VISIBLE_HISTORY = 5


def init_state() -> None:
    if HISTORY_KEY not in st.session_state:
        st.session_state[HISTORY_KEY] = []
    if PENDING_MISMATCH_KEY not in st.session_state:
        st.session_state[PENDING_MISMATCH_KEY] = None
    if VISIBLE_HISTORY_KEY not in st.session_state:
        st.session_state[VISIBLE_HISTORY_KEY] = DEFAULT_VISIBLE_HISTORY


def history() -> list[ChatTurn]:
    init_state()
    return st.session_state[HISTORY_KEY]


def add_turn(request: TranslationRequest, result: TranslationResult) -> None:
    history().append(
        ChatTurn(
            source_text=request.text,
            translation=result.translation,
            source_language=request.source_language,
            target_language=request.target_language,
            source_language_display=request.source_language_display,
            target_language_display=request.target_language_display,
            register=request.register,
            notes=result.notes,
            created_at=datetime.now(),
        )
    )


def add_unavailable_turn(
    request: TranslationRequest,
    detected_source_language_display: str,
) -> None:
    history().append(
        ChatTurn(
            source_text=request.text,
            translation=(
                "Traducción no disponible: el texto no parece corresponderse "
                f"con el idioma de origen seleccionado ({request.source_language_display})."
            ),
            source_language=request.source_language,
            target_language=request.target_language,
            source_language_display=request.source_language_display,
            target_language_display=request.target_language_display,
            register=request.register,
            notes=[
                (
                    "El idioma detectado parece ser "
                    f"{detected_source_language_display}, pero se mantuvo "
                    f"{request.source_language_display} como idioma de origen."
                )
            ],
            created_at=datetime.now(),
        )
    )


def clear_history() -> None:
    st.session_state[HISTORY_KEY] = []
    reset_visible_history()


def pending_mismatch() -> PendingLanguageMismatch | None:
    init_state()
    return st.session_state[PENDING_MISMATCH_KEY]


def set_pending_mismatch(pending: PendingLanguageMismatch) -> None:
    st.session_state[PENDING_MISMATCH_KEY] = pending


def clear_pending_mismatch() -> None:
    st.session_state[PENDING_MISMATCH_KEY] = None


def request_with_detected_source(pending: PendingLanguageMismatch) -> TranslationRequest:
    return replace(
        pending.request,
        source_language=pending.detected_source_language,
        source_language_display=pending.detected_source_language_display,
    )


def visible_history_count() -> int:
    init_state()
    return st.session_state[VISIBLE_HISTORY_KEY]


def show_more_history(step: int = DEFAULT_VISIBLE_HISTORY) -> None:
    st.session_state[VISIBLE_HISTORY_KEY] = visible_history_count() + step


def reset_visible_history() -> None:
    st.session_state[VISIBLE_HISTORY_KEY] = DEFAULT_VISIBLE_HISTORY
