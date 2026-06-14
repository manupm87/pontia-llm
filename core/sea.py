"""Acceso al estado del mar en Tenerife (oleaje) para el asistente.

Replica el patrón de ``core.weather``: consulta la API marina pública de
Open-Meteo y, si la llamada de red falla, recurre a una simulación determinista
de respaldo. Cada intento queda registrado en ``SEA_CALL_LOG`` para
observabilidad. Es útil para responder a dudas de playa, baño y surf.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime

import requests

from .config import TENERIFE_LATITUDE, TENERIFE_LONGITUDE

logger = logging.getLogger("asistente_tenerife.sea")

OPEN_METEO_MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"

# Registro de cada intento de consulta (observabilidad).
SEA_CALL_LOG: list[dict] = []


class SeaError(RuntimeError):
    """Error específico al obtener el estado del mar."""


def get_sea_conditions(
    date: str,
    *,
    latitude: float = TENERIFE_LATITUDE,
    longitude: float = TENERIFE_LONGITUDE,
    timeout: float = 10.0,
) -> dict:
    """Devuelve el estado del mar (oleaje) en Tenerife para una fecha dada.

    La ``date`` debe tener formato ``YYYY-MM-DD``; si no, se lanza ``ValueError``
    (lo captura la herramienta para avisar al usuario). Ante un fallo de red o
    parseo se usa una simulación determinista como respaldo. Cada intento se
    añade a ``SEA_CALL_LOG``.
    """
    # Validación de formato: este error SÍ se propaga.
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"Fecha inválida '{date}': se espera el formato YYYY-MM-DD."
        ) from exc

    start = time.perf_counter()
    error: str | None = None
    try:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "wave_height_max,wave_period_max",
            "timezone": "auto",
            "start_date": date,
            "end_date": date,
        }
        response = requests.get(OPEN_METEO_MARINE_URL, params=params, timeout=timeout)
        response.raise_for_status()
        result = _parse_response(date, response.json())
    except Exception as exc:  # noqa: BLE001 - cualquier fallo de red/parseo -> fallback
        error = str(exc)
        result = _simulated_sea(date)
    finally:
        elapsed = time.perf_counter() - start

    ok = error is None
    SEA_CALL_LOG.append(
        {
            "date": date,
            "ok": ok,
            "source": result["source"],
            "elapsed_s": elapsed,
            "error": error,
        }
    )

    if ok:
        logger.info("Estado del mar para %s desde %s en %.3fs.", date, result["source"], elapsed)
    else:
        logger.warning(
            "Fallo al consultar la API marina para %s (%s); usando simulación.",
            date,
            error,
        )

    return result


def _parse_response(date: str, payload: dict) -> dict:
    """Construye el diccionario de retorno a partir de la respuesta marina."""
    daily = payload["daily"]
    wave_height = float(daily["wave_height_max"][0])
    wave_period = float(daily["wave_period_max"][0])
    return {
        "date": date,
        "wave_height_max_m": wave_height,
        "wave_period_max_s": wave_period,
        "summary": _build_summary(wave_height),
        "source": "open-meteo",
    }


def _build_summary(wave_height: float) -> str:
    """Clasifica el estado del mar (escala Douglas simplificada) en español."""
    if wave_height < 0.5:
        return "Mar en calma, ideal para el baño"
    if wave_height < 1.25:
        return "Marejadilla, baño tranquilo"
    if wave_height < 2.5:
        return "Marejada, precaución al bañarse"
    return "Fuerte marejada, mar agitado; baño no recomendado"


def _simulated_sea(date: str) -> dict:
    """Estado del mar simulado determinista de respaldo para Tenerife.

    Usa el hash MD5 de la fecha para generar valores reproducibles y plausibles
    (oleaje suave habitual en las costas de la isla).
    """
    digest = hashlib.md5(date.encode("utf-8")).digest()
    # Altura entre 0.3 y 2.1 m; periodo entre 6 y 12 s.
    wave_height = round(0.3 + (digest[0] % 19) * 0.1, 1)
    wave_period = float(6 + digest[1] % 7)
    return {
        "date": date,
        "wave_height_max_m": wave_height,
        "wave_period_max_s": wave_period,
        "summary": _build_summary(wave_height),
        "source": "simulada (fallback)",
    }
