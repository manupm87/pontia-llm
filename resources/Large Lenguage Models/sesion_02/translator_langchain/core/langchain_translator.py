from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core.models import TranslationRequest


Provider = Literal["OpenAI", "Google Gemini"]

PROVIDER_MODEL_OPTIONS: dict[Provider, list[str]] = {
    "OpenAI": ["gpt-4.1-nano"],
    "Google Gemini": ["gemini-2.5-flash-lite"],
}

PROVIDER_TO_LANGCHAIN = {
    "OpenAI": "openai",
    "Google Gemini": "google_genai",
}


class TranslationError(RuntimeError):
    """Raised when the translation service cannot return a valid result."""


class LanguageDetection(BaseModel):
    detected_language: Literal[
        "English",
        "Spanish",
        "Italian",
        "French",
        "German",
        "Portuguese",
        "Catalan",
        "Galician",
        "Basque",
        "Dutch",
        "Chinese",
        "Japanese",
        "Korean",
        "Arabic",
        "Unknown",
    ] = Field(description="Dominant canonical English language name")


class LangChainTranslator:
    def __init__(self, provider: Provider, model: str, timeout: float) -> None:
        self._provider = provider
        self._model_name = model
        self._timeout = timeout
        self._model = init_chat_model(
            model,
            model_provider=PROVIDER_TO_LANGCHAIN[provider],
        )

    def detect_language(self, text: str) -> str:
        if not text.strip():
            raise TranslationError("El texto a traducir no puede estar vacío.")

        try:
            structured_model = self._model.with_structured_output(LanguageDetection)
            response = structured_model.invoke(
                [
                    SystemMessage(
                        content=(
                            "Detect the dominant language of the user text. "
                            "If the text is too short or ambiguous, choose the most likely language. "
                            "Use Unknown only when no reasonable language can be inferred."
                        )
                    ),
                    HumanMessage(content=text),
                ]
            )
        except Exception as exc:
            raise TranslationError(f"No se pudo detectar el idioma del texto: {exc}") from exc

        return response.detected_language

    def stream_translation(self, request: TranslationRequest) -> Iterator[str]:
        if not request.text.strip():
            raise TranslationError("El texto a traducir no puede estar vacío.")

        messages = [
            SystemMessage(content=_build_streaming_instructions(request)),
            HumanMessage(content=request.text),
        ]

        try:
            for chunk in self._model.stream(messages):
                content = chunk.content
                if isinstance(content, str):
                    yield content
        except Exception as exc:
            raise TranslationError(f"Se interrumpió la traducción en streaming: {exc}") from exc


def _build_streaming_instructions(request: TranslationRequest) -> str:
    formatting_rule = (
        "Preserve paragraph breaks, lists, punctuation, markdown and line breaks whenever possible."
        if request.preserve_formatting
        else "Do not preserve the original formatting if a clearer translation needs a different structure."
    )

    source_instruction = (
        f"Translate the user text from {request.source_language} to {request.target_language}."
        if request.source_language != "Auto-detect"
        else f"Detect the source language and translate the user text to {request.target_language}."
    )

    return f"""
You are a production-grade translation engine.

{source_instruction}
Use a {request.register} register.

Rules:
- Return only the translated text.
- Do not add explanations, labels, notes, markdown fences or surrounding quotes.
- Keep the meaning, nuance, names, numbers and domain-specific terms.
- {formatting_rule}
""".strip()

