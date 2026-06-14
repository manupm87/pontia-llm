from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


DEFAULT_MODEL = "gemini-2.5-flash-lite"


@dataclass(frozen=True)
class Settings:
    google_api_key: str | None
    model: str = DEFAULT_MODEL
    request_timeout: float = 45.0

    @property
    def has_api_key(self) -> bool:
        return bool(self.google_api_key)


def load_settings() -> Settings:
    load_dotenv()

    timeout = os.getenv("GOOGLE_TIMEOUT_SECONDS", "45")
    try:
        request_timeout = float(timeout)
    except ValueError:
        request_timeout = 45.0

    return Settings(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        model=os.getenv("GOOGLE_MODEL", DEFAULT_MODEL),
        request_timeout=request_timeout,
    )
