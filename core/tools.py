"""Herramientas LangChain que el LLM puede invocar.

Expone las herramientas (``@tool``) que el modelo Gemini puede llamar mediante
function calling:

- ``search_tourist_guide``: recuperación RAG sobre la guía oficial de Tenerife.
- ``get_weather``: previsión meteorológica para una fecha concreta.
- ``get_sea_conditions``: estado del mar (oleaje) para una fecha concreta.
- ``resolve_date``: traduce expresiones de fecha relativas a ``YYYY-MM-DD``.

El nombre que ve el modelo es el de la función; ``@tool`` deriva el JSON Schema
a partir de los type hints y el docstring en español. ``search_tourist_guide``
usa ``response_format="content_and_artifact"`` para devolver, además del texto,
las citas (fuentes y fotos) asociadas a esa llamada concreta, evitando depender
de estado compartido del RAG entre sesiones.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_core.tools import tool

from .dates import resolve_date as compute_date
from .sea import get_sea_conditions as fetch_sea
from .weather import get_weather as fetch_weather

if TYPE_CHECKING:  # Evita importar las dependencias pesadas del RAG al usar las tools.
    from .rag import TouristGuideRAG

# Instancia RAG compartida; se inyecta con ``set_rag_instance`` antes de usar
# la herramienta de búsqueda.
_rag: "TouristGuideRAG | None" = None


def set_rag_instance(rag: "TouristGuideRAG | None") -> None:
    """Asigna la instancia RAG global que usará ``search_tourist_guide``."""
    global _rag
    _rag = rag


@tool(response_format="content_and_artifact")
def search_tourist_guide(query: str) -> tuple[str, dict]:
    """Busca información en la guía oficial de Tenerife (TENERIFE.pdf).

    Devuelve fragmentos relevantes del documento con sus fuentes citadas
    (nombre, página y fragmento). Úsala para resolver dudas sobre lugares de
    interés, playas, rutas, gastronomía, cultura y cualquier otra información
    turística contenida en la guía.
    """
    if _rag is None:
        return "Error: el índice RAG no está inicializado.", {"sources": [], "images": []}
    result = _rag.retrieve(query)
    # El texto va al modelo; las citas viajan en el artifact (por llamada).
    return result["context"], {"sources": result["sources"], "images": result["images"]}


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


@tool
def get_sea_conditions(date: str) -> str:
    """Obtiene el estado del mar (oleaje) para una fecha en Tenerife.

    El parámetro ``date`` es la fecha solicitada en formato ``YYYY-MM-DD``.
    Devuelve la altura y el periodo máximos del oleaje y un resumen útil para
    saber si el mar está apto para el baño o el surf.
    """
    try:
        data = fetch_sea(date)
        return json.dumps(data, ensure_ascii=False)
    except ValueError as exc:
        return f"Error: fecha inválida ({exc}). Usa formato YYYY-MM-DD."


@tool
def resolve_date(expression: str) -> str:
    """Traduce una expresión de fecha relativa a formato ``YYYY-MM-DD``.

    Úsala SIEMPRE antes de ``get_weather`` o ``get_sea_conditions`` cuando el
    usuario mencione fechas relativas como 'hoy', 'mañana', 'este finde' o 'el
    miércoles'. Devuelve la fecha exacta para pasarla a esas herramientas.
    """
    try:
        return compute_date(expression)
    except ValueError:
        return (
            f"No se pudo interpretar '{expression}'. Pide al usuario una fecha "
            "concreta en formato YYYY-MM-DD."
        )


def get_tools() -> list:
    """Devuelve la lista de herramientas disponibles para el LLM."""
    return [search_tourist_guide, get_weather, get_sea_conditions, resolve_date]
