"""Tests de las herramientas LangChain expuestas al LLM (``core.tools``).

Comprueba que ``search_tourist_guide`` devuelve contexto + *artifact* con las
citas (sin depender de estado compartido del RAG), y que las herramientas de
fecha y estado del mar formatean su salida correctamente.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from unittest.mock import patch

from langchain_core.messages import ToolMessage

from core import tools


class _FakeRag:
    """RAG falso: devuelve un dict de recuperación fijo y registra la query."""

    def __init__(self) -> None:
        self.last_query: str | None = None

    def retrieve(self, query: str) -> dict:
        self.last_query = query
        return {
            "context": f"contexto para {query}",
            "sources": [{"source_name": "TENERIFE.pdf", "page": 3}],
            "images": [{"path": "/tmp/p04.png", "page": 3}],
        }


def test_search_returns_content_and_artifact_via_toolcall() -> None:
    fake = _FakeRag()
    tools.set_rag_instance(fake)
    call = {
        "name": "search_tourist_guide",
        "args": {"query": "playas del sur"},
        "id": "call_1",
        "type": "tool_call",
    }
    message = tools.search_tourist_guide.invoke(call)

    assert isinstance(message, ToolMessage)
    assert message.content == "contexto para playas del sur"
    # Las citas viajan en el artifact, por llamada (no por estado compartido).
    assert message.artifact["sources"][0]["page"] == 3
    assert message.artifact["images"][0]["path"] == "/tmp/p04.png"
    assert fake.last_query == "playas del sur"


def test_search_without_rag_returns_error_artifact() -> None:
    tools.set_rag_instance(None)
    call = {
        "name": "search_tourist_guide",
        "args": {"query": "x"},
        "id": "call_2",
        "type": "tool_call",
    }
    message = tools.search_tourist_guide.invoke(call)
    assert "no está inicializado" in message.content.lower()
    assert message.artifact == {"sources": [], "images": []}


def test_resolve_date_tool_returns_iso() -> None:
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    assert tools.resolve_date.invoke({"expression": "mañana"}) == tomorrow


def test_resolve_date_tool_handles_bad_input() -> None:
    out = tools.resolve_date.invoke({"expression": "el siglo que viene"})
    assert "YYYY-MM-DD" in out


def test_get_sea_conditions_tool_returns_json() -> None:
    payload = {
        "date": "2026-06-20",
        "wave_height_max_m": 0.5,
        "wave_period_max_s": 7.0,
        "summary": "Mar en calma",
        "source": "open-meteo",
    }
    with patch("core.tools.fetch_sea", return_value=payload):
        out = tools.get_sea_conditions.invoke({"date": "2026-06-20"})
    assert json.loads(out)["wave_height_max_m"] == 0.5


def test_get_sea_conditions_tool_validates_date() -> None:
    out = tools.get_sea_conditions.invoke({"date": "nope"})
    assert "YYYY-MM-DD" in out


def test_get_weather_tool_returns_json() -> None:
    payload = {
        "date": "2026-06-20",
        "temp_max_c": 28.0,
        "temp_min_c": 21.0,
        "precipitation_mm": 0.0,
        "summary": "Soleado",
        "source": "open-meteo",
    }
    with patch("core.tools.fetch_weather", return_value=payload):
        out = tools.get_weather.invoke({"date": "2026-06-20"})
    assert json.loads(out)["temp_max_c"] == 28.0


def test_get_weather_tool_validates_date() -> None:
    out = tools.get_weather.invoke({"date": "nope"})
    assert "YYYY-MM-DD" in out


def test_get_tools_exposes_all_four() -> None:
    names = {t.name for t in tools.get_tools()}
    assert names == {
        "search_tourist_guide",
        "get_weather",
        "get_sea_conditions",
        "resolve_date",
    }
