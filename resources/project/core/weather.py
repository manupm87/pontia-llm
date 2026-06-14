"""Acceso a la previsión meteorológica de Tenerife.

Este módulo consulta la API pública de Open-Meteo para obtener la previsión
diaria. Si la llamada de red falla por cualquier motivo, recurre a una
simulación determinista como mecanismo de respaldo, de modo que el asistente
siempre disponga de una respuesta. Cada intento queda registrado en
``WEATHER_CALL_LOG`` para facilitar la observabilidad.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime

import requests

from .config import TENERIFE_LATITUDE, TENERIFE_LONGITUDE

logger = logging.getLogger("asistente_tenerife.weather")

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Registro de cada intento de consulta (observabilidad).
WEATHER_CALL_LOG: list[dict] = []


class WeatherError(RuntimeError):
    """Error específico al obtener la previsión meteorológica."""


def get_weather(
    date: str,
    *,
    latitude: float = TENERIFE_LATITUDE,
    longitude: float = TENERIFE_LONGITUDE,
    timeout: float = 10.0,
) -> dict:
    """Devuelve la previsión meteorológica de Tenerife para una fecha dada.

    La ``date`` debe tener formato ``YYYY-MM-DD``; si no, se lanza
    ``ValueError`` (lo captura la herramienta para avisar al usuario). Si la
    llamada a Open-Meteo falla por red o parseo, se usa una previsión simulada
    determinista como respaldo. Cualquier intento se añade a
    ``WEATHER_CALL_LOG``.
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
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "auto",
            "start_date": date,
            "end_date": date,
        }
        response = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
        response.raise_for_status()
        result = _parse_response(date, response.json())
    except Exception as exc:  # noqa: BLE001 - cualquier fallo de red/parseo -> fallback
        error = str(exc)
        result = _simulated_weather(date)
    finally:
        elapsed = time.perf_counter() - start

    ok = error is None
    record = {
        "date": date,
        "ok": ok,
        "source": result["source"],
        "elapsed_s": elapsed,
        "error": error,
    }
    WEATHER_CALL_LOG.append(record)

    if ok:
        logger.info(
            "Previsión obtenida para %s desde %s en %.3fs.",
            date,
            result["source"],
            elapsed,
        )
    else:
        logger.warning(
            "Fallo al consultar Open-Meteo para %s (%s); usando previsión simulada.",
            date,
            error,
        )

    return result


def _parse_response(date: str, payload: dict) -> dict:
    """Construye el diccionario de retorno a partir de la respuesta de Open-Meteo."""
    daily = payload["daily"]
    temp_max = float(daily["temperature_2m_max"][0])
    temp_min = float(daily["temperature_2m_min"][0])
    precipitation = float(daily["precipitation_sum"][0])
    return {
        "date": date,
        "temperature_max_c": temp_max,
        "temperature_min_c": temp_min,
        "precipitation_mm": precipitation,
        "summary": _build_summary(temp_max, precipitation),
        "source": "open-meteo",
    }


def _build_summary(temperature_max: float, precipitation: float) -> str:
    """Genera un resumen en español a partir de la temperatura y la lluvia."""
    if precipitation >= 5.0:
        condition = "Lluvioso"
    elif precipitation > 0.0:
        condition = "Chubascos ocasionales"
    elif temperature_max >= 26.0:
        condition = "Soleado"
    else:
        condition = "Parcialmente nublado"
    return f"{condition}, máx {round(temperature_max)}°C"


def _simulated_weather(date: str) -> dict:
    """Previsión simulada determinista de respaldo para Tenerife.

    Usa el hash MD5 de la fecha para generar valores reproducibles y plausibles
    (temperaturas típicas de la isla, entre 18 y 30°C).
    """
    digest = hashlib.md5(date.encode("utf-8")).digest()

    # Máxima entre 22 y 30°C; mínima 4-7°C por debajo de la máxima.
    temp_max = 22.0 + digest[0] % 9  # 22..30
    spread = 4 + digest[1] % 4  # 4..7
    temp_min = max(18.0, temp_max - spread)
    # Lluvia: la mayoría de días sin precipitación.
    precipitation = round((digest[2] % 6) * 0.8, 1) if digest[3] % 4 == 0 else 0.0

    return {
        "date": date,
        "temperature_max_c": float(temp_max),
        "temperature_min_c": float(temp_min),
        "precipitation_mm": float(precipitation),
        "summary": _build_summary(temp_max, precipitation),
        "source": "simulada (fallback)",
    }
