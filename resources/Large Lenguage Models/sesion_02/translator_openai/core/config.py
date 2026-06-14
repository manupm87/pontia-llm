from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


DEFAULT_MODEL = "gpt-4.1-nano"


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    model: str = DEFAULT_MODEL
    request_timeout: float = 45.0

    @property
    def has_api_key(self) -> bool:
        return bool(self.openai_api_key)


def load_settings() -> Settings:
    load_dotenv()

    timeout = os.getenv("OPENAI_TIMEOUT_SECONDS", "45")
    try:
        request_timeout = float(timeout)
    except ValueError:
        request_timeout = 45.0

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        request_timeout=request_timeout,
    )

