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

import requests  # noqa: F401 - reexportado para que los tests puedan parchear ``requests.get``

from .config import TENERIFE_LATITUDE, TENERIFE_LONGITUDE
from .meteo import fetch_with_fallback

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
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "auto",
        "start_date": date,
        "end_date": date,
    }
    return fetch_with_fallback(
        date,
        url=OPEN_METEO_URL,
        params=params,
        parse_fn=_parse_response,
        simulate_fn=_simulated_weather,
        call_log=WEATHER_CALL_LOG,
        logger=logger,
        label="Previsión meteorológica",
        timeout=timeout,
    )


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
    digest = hashlib.md5(date.encode("utf-8"), usedforsecurity=False).digest()

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
