from __future__ import annotations

import json
from collections.abc import Iterator

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from core.models import TranslationRequest, TranslationResult


TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {
        "translation": {"type": "string"},
        "detected_source_language": {"type": "string"},
        "source_language": {"type": "string"},
        "target_language": {"type": "string"},
        "notes": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "translation",
        "detected_source_language",
        "source_language",
        "target_language",
        "notes",
    ]
}

LANGUAGE_DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "detected_language": {
            "type": "string",
            "enum": [
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
            ],
        }
    },
    "required": ["detected_language"]
}


class TranslationError(RuntimeError):
    """Raised when the translation service cannot return a valid result."""


class GeminiTranslator:
    def __init__(self, api_key: str, model: str, timeout: float) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._timeout = timeout

    def translate(self, request: TranslationRequest) -> TranslationResult:
        if not request.text.strip():
            raise TranslationError("El texto a traducir no puede estar vacío.")

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=request.text,
                config=genai_types.GenerateContentConfig(
                    system_instruction=_build_instructions(request),
                    response_mime_type="application/json",
                    response_schema=TRANSLATION_SCHEMA,
                ),
            )
        except genai_errors.APIError as exc:
            raise TranslationError(f"No se pudo completar la llamada a Gemini: {exc}") from exc

        try:
            payload = json.loads(response.text or "{}")
        except json.JSONDecodeError as exc:
            raise TranslationError("Gemini devolvió una respuesta que no era JSON válido.") from exc

        return TranslationResult(
            translation=payload["translation"],
            detected_source_language=payload["detected_source_language"],
            source_language=payload["source_language"],
            target_language=payload["target_language"],
            notes=payload["notes"],
        )

    def detect_language(self, text: str) -> str:
        if not text.strip():
            raise TranslationError("El texto a traducir no puede estar vacío.")

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=text,
                config=genai_types.GenerateContentConfig(
                    system_instruction=(
                        "Detect the dominant language of the user text. "
                        "Return only data matching the requested JSON schema. "
                        "If the text is too short or ambiguous, choose the most likely language. "
                        "Use Unknown only when no reasonable language can be inferred."
                    ),
                    response_mime_type="application/json",
                    response_schema=LANGUAGE_DETECTION_SCHEMA,
                ),
            )
        except genai_errors.APIError as exc:
            raise TranslationError(f"No se pudo detectar el idioma del texto: {exc}") from exc

        try:
            payload = json.loads(response.text or "{}")
        except json.JSONDecodeError as exc:
            raise TranslationError("Gemini devolvió una detección de idioma no válida.") from exc

        return payload["detected_language"]

    def stream_translation(self, request: TranslationRequest) -> Iterator[str]:
        if not request.text.strip():
            raise TranslationError("El texto a traducir no puede estar vacío.")

        try:
            stream = self._client.models.generate_content_stream(
                model=self._model,
                contents=request.text,
                config=genai_types.GenerateContentConfig(
                    system_instruction=_build_streaming_instructions(request),
                ),
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except genai_errors.APIError as exc:
            raise TranslationError(f"Se interrumpió la traducción en streaming: {exc}") from exc


def _build_instructions(request: TranslationRequest) -> str:
    formatting_rule = (
        "Preserve paragraph breaks, lists, punctuation, markdown and line breaks whenever possible."
        if request.preserve_formatting
        else "Do not preserve the original formatting if a clearer translation needs a different structure."
    )

    source_instruction = (
        f"Detect the source language of the user text and translate it to {request.target_language}."
        if request.source_language == "Auto-detect"
        else f"Translate the user text from {request.source_language} to {request.target_language}."
    )

    return f"""
You are a production-grade translation engine.

{source_instruction}
Use a {request.register} register.

Rules:
- Return only data matching the requested JSON schema.
- Do not add explanations outside the JSON object.
- Keep the meaning, nuance, names, numbers and domain-specific terms.
- Always set detected_source_language to the canonical English language name you detect.
- If the source language selected by the user does not match the text, translate anyway and report the detected source language.
- If the source language is auto-detect, set source_language to the detected canonical English language name.
- Add concise notes only when there is ambiguity, idiom handling, tone adaptation or source-language mismatch.
- If there are no relevant notes, return an empty notes array.
- {formatting_rule}
""".strip()


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

