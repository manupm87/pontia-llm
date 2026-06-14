from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from core.categories import CategoryName


@dataclass(frozen=True)
class IngestionConfig:
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int
    category_filter: CategoryName | None
    use_query_analysis: bool


@dataclass(frozen=True)
class IndexFingerprint:
    source_label: str
    source_size: int
    source_mtime: float
    chunk_size: int
    chunk_overlap: int

    def as_key(self) -> tuple:
        return (
            self.source_label,
            self.source_size,
            self.source_mtime,
            self.chunk_size,
            self.chunk_overlap,
        )


@dataclass(frozen=True)
class ChunkSummary:
    chunk_id: int
    page: int | None
    category: CategoryName
    char_count: int


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    page: int | None
    source_name: str
    category: CategoryName
    content: str


@dataclass(frozen=True)
class Citation:
    source_name: str
    page: int | None
    chunk_id: int
    category: CategoryName


@dataclass(frozen=True)
class AnalyzedQuery:
    query: str
    category: CategoryName


@dataclass(frozen=True)
class AskRequest:
    question: str
    retrieval: RetrievalConfig


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    chunks: tuple[RetrievedChunk, ...]
    analyzed_query: AnalyzedQuery | None
    effective_query: str


AnswerStatus = Literal["answered", "no_context"]


@dataclass(frozen=True)
class ChatTurn:
    question: str
    answer: str
    chunks: tuple[RetrievedChunk, ...]
    citations: tuple[Citation, ...]
    analyzed_query: AnalyzedQuery | None
    effective_query: str
    retrieval: RetrievalConfig
    status: AnswerStatus
    created_at: datetime = field(default_factory=datetime.now)
