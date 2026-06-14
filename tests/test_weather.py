"""Tests de ``core.weather.get_weather`` (previsión meteorológica de Tenerife).

Se simula la respuesta de la API de Open-Meteo y se comprueba el camino de
éxito, las bandas del resumen, el respaldo simulado ante fallos de red y la
validación de fechas.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from core import weather
from core.weather import get_weather


@pytest.fixture(autouse=True)
def _clear_log():
    weather.WEATHER_CALL_LOG.clear()
    yield
    weather.WEATHER_CALL_LOG.clear()


def _fake_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def test_invalid_date_raises_value_error() -> None:
    with pytest.raises(ValueError):
        get_weather("14-06-2026")


def test_success_path_parses_payload() -> None:
    payload = {
        "daily": {
            "temperature_2m_max": [27.0],
            "temperature_2m_min": [21.0],
            "precipitation_sum": [0.0],
        }
    }
    with patch("core.weather.requests.get", return_value=_fake_response(payload)) as mock_get:
        result = get_weather("2026-06-20")

    mock_get.assert_called_once()
    assert result["date"] == "2026-06-20"
    assert result["temperature_max_c"] == 27.0
    assert result["temperature_min_c"] == 21.0
    assert result["precipitation_mm"] == 0.0
    assert result["source"] == "open-meteo"
    # Temperatura alta y sin lluvia => soleado.
    assert "soleado" in result["summary"].lower()
    assert weather.WEATHER_CALL_LOG[-1]["ok"] is True


@pytest.mark.parametrize(
    "temperature_max, precipitation, keyword",
    [
        (24.0, 8.0, "lluvioso"),
        (24.0, 1.0, "chubascos"),
        (28.0, 0.0, "soleado"),
        (22.0, 0.0, "parcialmente nublado"),
    ],
)
def test_summary_bands(temperature_max: float, precipitation: float, keyword: str) -> None:
    from core.weather import _build_summary

    assert keyword in _build_summary(temperature_max, precipitation).lower()


def test_simulation_within_documented_ranges() -> None:
    from core.weather import _simulated_weather

    for day in range(1, 29):
        result = _simulated_weather(f"2026-06-{day:02d}")
        assert 22.0 <= result["temperature_max_c"] <= 30.0
        assert 18.0 <= result["temperature_min_c"] <= result["temperature_max_c"]
        assert result["precipitation_mm"] >= 0.0


def test_network_failure_falls_back_to_simulation() -> None:
    with patch("core.weather.requests.get", side_effect=requests.RequestException("boom")):
        result = get_weather("2026-06-20")
    assert result["source"] == "simulada (fallback)"
    assert "temperature_max_c" in result
    assert weather.WEATHER_CALL_LOG[-1]["ok"] is False
    assert weather.WEATHER_CALL_LOG[-1]["error"] == "boom"


def test_simulation_is_deterministic() -> None:
    with patch("core.weather.requests.get", side_effect=requests.RequestException("boom")):
        a = get_weather("2026-06-20")
        b = get_weather("2026-06-20")
    assert a["temperature_max_c"] == b["temperature_max_c"]
    assert a["precipitation_mm"] == b["precipitation_mm"]


def test_http_200_error_payload_falls_back_to_simulation() -> None:
    # Open-Meteo puede responder HTTP 200 con un cuerpo de error.
    payload = {"error": True, "reason": "fechas fuera de rango"}
    with patch("core.weather.requests.get", return_value=_fake_response(payload)):
        result = get_weather("2026-06-20")
    assert result["source"] == "simulada (fallback)"
    assert weather.WEATHER_CALL_LOG[-1]["ok"] is False
    assert "fechas fuera de rango" in weather.WEATHER_CALL_LOG[-1]["error"]
