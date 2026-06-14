from __future__ import annotations

import streamlit as st

from core.models import ChatTurn


HISTORY_KEY = "rag_history"
VISIBLE_HISTORY_KEY = "rag_visible_history"
INDEX_HANDLE_KEY = "rag_index_handle"
INDEX_FINGERPRINT_KEY = "rag_index_fingerprint"
INDEX_SOURCE_KEY = "rag_index_source_label"
AUTOBUILD_ATTEMPTED_KEY = "rag_autobuild_attempted"
DEFAULT_VISIBLE_HISTORY = 4


def init_state() -> None:
    defaults = {
        HISTORY_KEY: [],
        VISIBLE_HISTORY_KEY: DEFAULT_VISIBLE_HISTORY,
        INDEX_HANDLE_KEY: None,
        INDEX_FINGERPRINT_KEY: None,
        INDEX_SOURCE_KEY: None,
        AUTOBUILD_ATTEMPTED_KEY: False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def history() -> list[ChatTurn]:
    init_state()
    return st.session_state[HISTORY_KEY]


def add_turn(turn: ChatTurn) -> None:
    history().append(turn)


def clear_history() -> None:
    st.session_state[HISTORY_KEY] = []
    st.session_state[VISIBLE_HISTORY_KEY] = DEFAULT_VISIBLE_HISTORY


def visible_history_count() -> int:
    init_state()
    return st.session_state[VISIBLE_HISTORY_KEY]


def show_more_history(step: int = DEFAULT_VISIBLE_HISTORY) -> None:
    st.session_state[VISIBLE_HISTORY_KEY] = visible_history_count() + step


def set_index_handle(handle, fingerprint, source_label: str) -> None:
    st.session_state[INDEX_HANDLE_KEY] = handle
    st.session_state[INDEX_FINGERPRINT_KEY] = fingerprint
    st.session_state[INDEX_SOURCE_KEY] = source_label


def index_handle():
    init_state()
    return st.session_state[INDEX_HANDLE_KEY]


def index_fingerprint():
    init_state()
    return st.session_state[INDEX_FINGERPRINT_KEY]


def index_source_label() -> str | None:
    init_state()
    return st.session_state[INDEX_SOURCE_KEY]


def mark_autobuild_attempted() -> None:
    st.session_state[AUTOBUILD_ATTEMPTED_KEY] = True


def autobuild_attempted() -> bool:
    init_state()
    return bool(st.session_state[AUTOBUILD_ATTEMPTED_KEY])


def reset_autobuild_flag() -> None:
    st.session_state[AUTOBUILD_ATTEMPTED_KEY] = False
