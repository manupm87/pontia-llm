from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


DEFAULT_PROVIDER = "OpenAI"
DEFAULT_MODEL = "gpt-4.1-nano"


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    google_api_key: str | None
    default_provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    request_timeout: float = 45.0

    def has_api_key(self, provider: str) -> bool:
        if provider == "OpenAI":
            return bool(self.openai_api_key)
        if provider == "Google Gemini":
            return bool(self.google_api_key)
        return False


def load_settings() -> Settings:
    load_dotenv()

    timeout = os.getenv("LANGCHAIN_TIMEOUT_SECONDS", "45")
    try:
        request_timeout = float(timeout)
    except ValueError:
        request_timeout = 45.0

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        default_provider=os.getenv("LANGCHAIN_PROVIDER", DEFAULT_PROVIDER),
        model=os.getenv("LANGCHAIN_MODEL", DEFAULT_MODEL),
        request_timeout=request_timeout,
    )
