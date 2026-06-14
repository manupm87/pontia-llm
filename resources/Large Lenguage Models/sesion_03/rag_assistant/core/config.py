from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_GENERATION_MODEL = "gemini-2.5-flash-lite"
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_TOP_K = 3
DEFAULT_REQUEST_TIMEOUT = 45.0

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PATH = (APP_ROOT.parent / "inputs" / "company_docs").resolve()


@dataclass(frozen=True)
class Settings:
    google_api_key: str | None
    generation_model: str = DEFAULT_GENERATION_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    default_pdf_path: Path = DEFAULT_SOURCE_PATH
    default_top_k: int = DEFAULT_TOP_K
    default_chunk_size: int = DEFAULT_CHUNK_SIZE
    default_chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT

    @property
    def has_api_key(self) -> bool:
        return bool(self.google_api_key)


def load_settings() -> Settings:
    load_dotenv()

    source_path_str = os.getenv("RAG_DEFAULT_SOURCE_PATH") or os.getenv("RAG_DEFAULT_PDF_PATH")
    source_path = (
        Path(source_path_str).expanduser().resolve()
        if source_path_str
        else DEFAULT_SOURCE_PATH
    )

    return Settings(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        generation_model=os.getenv("RAG_GENERATION_MODEL", DEFAULT_GENERATION_MODEL),
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        default_pdf_path=source_path,
        default_top_k=_safe_int(os.getenv("RAG_DEFAULT_TOP_K"), DEFAULT_TOP_K),
        default_chunk_size=_safe_int(os.getenv("RAG_DEFAULT_CHUNK_SIZE"), DEFAULT_CHUNK_SIZE),
        default_chunk_overlap=_safe_int(
            os.getenv("RAG_DEFAULT_CHUNK_OVERLAP"), DEFAULT_CHUNK_OVERLAP
        ),
        request_timeout=_safe_float(os.getenv("RAG_TIMEOUT_SECONDS"), DEFAULT_REQUEST_TIMEOUT),
    )


def _safe_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _safe_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default
