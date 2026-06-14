"""Tests de ``core.sea.get_sea_conditions`` (estado del mar en Tenerife).

Se simula la respuesta de la API marina de Open-Meteo y se comprueba el camino
de éxito, el respaldo simulado ante fallos de red y la validación de fechas.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from core import sea
from core.sea import get_sea_conditions


@pytest.fixture(autouse=True)
def _clear_log():
    sea.SEA_CALL_LOG.clear()
    yield
    sea.SEA_CALL_LOG.clear()


def _fake_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def test_invalid_date_raises_value_error() -> None:
    with pytest.raises(ValueError):
        get_sea_conditions("14-06-2026")


def test_success_path_parses_marine_payload() -> None:
    payload = {
        "daily": {
            "wave_height_max": [0.4],
            "wave_period_max": [7.0],
        }
    }
    with patch("core.sea.requests.get", return_value=_fake_response(payload)) as mock_get:
        result = get_sea_conditions("2026-06-20")

    mock_get.assert_called_once()
    assert result["date"] == "2026-06-20"
    assert result["wave_height_max_m"] == 0.4
    assert result["wave_period_max_s"] == 7.0
    assert result["source"] == "open-meteo"
    # Oleaje bajo => mar en calma y apto para el baño.
    assert "calma" in result["summary"].lower()
    assert sea.SEA_CALL_LOG[-1]["ok"] is True


def test_high_waves_summary_warns() -> None:
    payload = {"daily": {"wave_height_max": [3.0], "wave_period_max": [11.0]}}
    with patch("core.sea.requests.get", return_value=_fake_response(payload)):
        result = get_sea_conditions("2026-06-20")
    # Oleaje fuerte: el resumen debe reflejar mar agitado/marejada.
    assert any(word in result["summary"].lower() for word in ("marejada", "agitado", "fuerte"))


@pytest.mark.parametrize(
    "wave_height, keyword",
    [(0.3, "calma"), (0.8, "marejadilla"), (1.8, "marejada"), (3.0, "fuerte")],
)
def test_summary_bands(wave_height: float, keyword: str) -> None:
    from core.sea import _build_summary

    assert keyword in _build_summary(wave_height).lower()


def test_simulation_within_documented_ranges() -> None:
    from core.sea import _simulated_sea

    for day in range(1, 29):
        result = _simulated_sea(f"2026-06-{day:02d}")
        assert 0.3 <= result["wave_height_max_m"] <= 2.1
        assert 6 <= result["wave_period_max_s"] <= 12


def test_network_failure_falls_back_to_simulation() -> None:
    with patch("core.sea.requests.get", side_effect=requests.RequestException("boom")):
        result = get_sea_conditions("2026-06-20")
    assert result["source"] == "simulada (fallback)"
    assert "wave_height_max_m" in result
    assert sea.SEA_CALL_LOG[-1]["ok"] is False
    assert sea.SEA_CALL_LOG[-1]["error"] == "boom"


def test_simulation_is_deterministic() -> None:
    with patch("core.sea.requests.get", side_effect=requests.RequestException("boom")):
        a = get_sea_conditions("2026-06-20")
        b = get_sea_conditions("2026-06-20")
    assert a["wave_height_max_m"] == b["wave_height_max_m"]


def test_http_200_error_payload_falls_back_to_simulation() -> None:
    # Open-Meteo puede responder HTTP 200 con un cuerpo de error.
    payload = {"error": True, "reason": "fechas fuera de rango"}
    with patch("core.sea.requests.get", return_value=_fake_response(payload)):
        result = get_sea_conditions("2026-06-20")
    assert result["source"] == "simulada (fallback)"
    assert sea.SEA_CALL_LOG[-1]["ok"] is False
    assert "fechas fuera de rango" in sea.SEA_CALL_LOG[-1]["error"]
