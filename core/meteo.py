"""Orquestación compartida de las consultas meteorológicas a Open-Meteo.

Tanto la previsión del tiempo (``core.weather``) como el estado del mar
(``core.sea``) siguen el mismo patrón: validar la fecha, llamar a la API pública
de Open-Meteo, medir el tiempo, y, ante cualquier fallo de red o parseo, recurrir
a una simulación determinista de respaldo. Este módulo concentra ese flujo común
para evitar la duplicación; cada módulo aporta su URL, parámetros, parser y
simulador.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime

import requests

# Errores tratados como fallo recuperable -> se usa la simulación de respaldo.
# Incluye ``RuntimeError`` para los payloads de error con HTTP 200 de Open-Meteo.
FALLBACK_ERRORS = (
    requests.RequestException,
    KeyError,
    IndexError,
    ValueError,
    TypeError,
    RuntimeError,
)


def _validate_date(date: str) -> None:
    """Valida que la fecha tenga formato ``YYYY-MM-DD``; si no, lanza ``ValueError``."""
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"Fecha inválida '{date}': se espera el formato YYYY-MM-DD."
        ) from exc


def fetch_with_fallback(
    date: str,
    *,
    url: str,
    params: dict,
    parse_fn: Callable[[str, dict], dict],
    simulate_fn: Callable[[str], dict],
    call_log: list[dict],
    logger: logging.Logger,
    label: str,
    timeout: float,
) -> dict:
    """Consulta Open-Meteo y recurre a una simulación determinista si falla.

    Valida la ``date`` (``ValueError`` si el formato no es ``YYYY-MM-DD``),
    cronometra la llamada, registra el intento en ``call_log`` y devuelve el
    diccionario producido por ``parse_fn`` (éxito) o por ``simulate_fn``
    (respaldo). Open-Meteo puede responder con HTTP 200 y un cuerpo
    ``{"error": true, "reason": ...}``; ese caso se detecta explícitamente y se
    trata como fallo para activar el respaldo. El ``label`` solo se usa en los
    mensajes de log.
    """
    # Validación de formato: este error SÍ se propaga.
    _validate_date(date)

    start = time.perf_counter()
    error: str | None = None
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        # Open-Meteo puede devolver HTTP 200 con un cuerpo de error.
        if isinstance(payload, dict) and payload.get("error"):
            reason = payload.get("reason", "motivo no especificado")
            raise RuntimeError(f"Open-Meteo devolvió un error: {reason}")
        result = parse_fn(date, payload)
    except FALLBACK_ERRORS as exc:
        error = str(exc)
        result = simulate_fn(date)
    finally:
        elapsed = time.perf_counter() - start

    ok = error is None
    call_log.append(
        {
            "date": date,
            "ok": ok,
            "source": result["source"],
            "elapsed_s": elapsed,
            "error": error,
        }
    )

    if ok:
        logger.info("%s para %s desde %s en %.3fs.", label, date, result["source"], elapsed)
    else:
        logger.warning(
            "Fallo al consultar Open-Meteo (%s) para %s (%s); usando simulación.",
            label,
            date,
            error,
        )

    return result
