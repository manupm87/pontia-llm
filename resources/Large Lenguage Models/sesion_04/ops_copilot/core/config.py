from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"

DEFAULT_GENERATION_MODEL = "gpt-4.1-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TOP_K = 4
DEFAULT_REQUEST_TIMEOUT = 45.0


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    generation_model: str = DEFAULT_GENERATION_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    default_top_k: int = DEFAULT_TOP_K
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT

    @property
    def has_api_key(self) -> bool:
        return bool(self.openai_api_key)


def load_settings() -> Settings:
    load_dotenv(APP_ROOT / ".env")
    load_dotenv()

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        generation_model=os.getenv("OPENAI_GENERATION_MODEL", DEFAULT_GENERATION_MODEL),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        default_top_k=_safe_int(os.getenv("OPS_DEFAULT_TOP_K"), DEFAULT_TOP_K),
        request_timeout=_safe_float(os.getenv("OPENAI_TIMEOUT_SECONDS"), DEFAULT_REQUEST_TIMEOUT),
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
