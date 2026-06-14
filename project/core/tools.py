"""Herramientas LangChain que el LLM puede invocar.

Expone dos herramientas (``@tool``) que el modelo Gemini puede llamar mediante
function calling:

- ``search_tourist_guide``: recuperación RAG sobre la guía oficial de Tenerife.
- ``get_weather``: previsión meteorológica para una fecha concreta.

El nombre que ve el modelo es el de la función; ``@tool`` deriva el JSON Schema
a partir de los type hints y el docstring en español.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from .rag import TouristGuideRAG
from .weather import get_weather as fetch_weather

# Instancia RAG compartida; se inyecta con ``set_rag_instance`` antes de usar
# la herramienta de búsqueda.
_rag: TouristGuideRAG | None = None


def set_rag_instance(rag: TouristGuideRAG) -> None:
    """Asigna la instancia RAG global que usará ``search_tourist_guide``."""
    global _rag
    _rag = rag


@tool
def search_tourist_guide(query: str) -> str:
    """Busca información en la guía oficial de Tenerife (TENERIFE.pdf).

    Devuelve fragmentos relevantes del documento con sus fuentes citadas
    (nombre, página y fragmento). Úsala para resolver dudas sobre lugares de
    interés, playas, rutas, gastronomía, cultura y cualquier otra información
    turística contenida en la guía.
    """
    if _rag is None:
        return "Error: el índice RAG no está inicializado."
    return _rag.retrieve(query)["context"]


@tool
def get_weather(date: str) -> str:
    """Obtiene la previsión meteorológica para una fecha en Tenerife.

    El parámetro ``date`` es la fecha solicitada en formato ``YYYY-MM-DD``.
    Devuelve un resumen con temperaturas máxima y mínima y precipitación.
    """
    try:
        data = fetch_weather(date)
        return json.dumps(data, ensure_ascii=False)
    except ValueError as exc:
        return f"Error: fecha inválida ({exc}). Usa formato YYYY-MM-DD."


def get_tools() -> list:
    """Devuelve la lista de herramientas disponibles para el LLM."""
    return [search_tourist_guide, get_weather]
