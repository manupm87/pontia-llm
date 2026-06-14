from __future__ import annotations

import streamlit as st

from core.models import ChatTurn, DraftStatus, ExpenseDraft


HISTORY_KEY = "ops_history"
VISIBLE_HISTORY_KEY = "ops_visible_history"
DRAFTS_KEY = "ops_expense_drafts"
DEFAULT_VISIBLE_HISTORY = 5


def init_state() -> None:
    st.session_state.setdefault(HISTORY_KEY, [])
    st.session_state.setdefault(VISIBLE_HISTORY_KEY, DEFAULT_VISIBLE_HISTORY)
    st.session_state.setdefault(DRAFTS_KEY, {})


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


def drafts() -> dict[str, ExpenseDraft]:
    init_state()
    return st.session_state[DRAFTS_KEY]


def update_draft_status(draft_id: str, status: DraftStatus) -> None:
    current = drafts().get(draft_id)
    if current is not None:
        current.status = status
