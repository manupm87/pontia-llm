"""Normalización de texto compartida por varios módulos del núcleo.

Centraliza la limpieza de texto que antes estaba triplicada (guardarraíles,
fechas, evaluación y emparejado de fotos): normaliza a NFKD, elimina los acentos
combinantes, pasa a minúsculas, colapsa los espacios y recorta los extremos. Así
todos los módulos comparan cadenas con exactamente las mismas reglas.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_text(value: str) -> str:
    """Normaliza el texto para comparaciones laxas (sin acentos ni mayúsculas).

    Pasa a minúsculas, descompone en NFKD para separar los acentos, elimina los
    caracteres combinantes, colapsa los espacios consecutivos en uno y recorta
    los espacios de los extremos.
    """
    text = unicodedata.normalize("NFKD", (value or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip()
