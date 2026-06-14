from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


Register = Literal["neutral", "formal", "informal", "technical", "marketing"]


@dataclass(frozen=True)
class TranslationRequest:
    text: str
    source_language: str
    target_language: str
    source_language_display: str
    target_language_display: str
    register: Register = "neutral"
    preserve_formatting: bool = True


@dataclass(frozen=True)
class TranslationResult:
    translation: str
    detected_source_language: str
    source_language: str
    target_language: str
    notes: list[str]


@dataclass(frozen=True)
class ChatTurn:
    source_text: str
    translation: str
    source_language: str
    target_language: str
    source_language_display: str
    target_language_display: str
    register: str
    notes: list[str]
    created_at: datetime


@dataclass(frozen=True)
class PendingLanguageMismatch:
    request: TranslationRequest
    detected_source_language: str
    detected_source_language_display: str
