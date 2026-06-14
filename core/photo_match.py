"""Emparejado de fotos con sus menciones en el texto de la respuesta.

Lógica pura (sin Streamlit ni E/S) que decide en qué punto de la respuesta se
intercala cada foto: empareja las palabras distintivas del pie de cada foto
(normalmente el nombre del lugar) con la línea del texto que lo nombra. Se
mantiene aparte de la interfaz para poder probarla.
"""

from __future__ import annotations

import re
import unicodedata

# Palabras genéricas de tipo de lugar: no sirven para emparejar (toda foto de
# "playa" coincidiría con cualquier mención de "playa").
PLACE_STOPWORDS = {
    "playa", "playas", "mirador", "miradores", "parque", "punta", "calle",
    "plaza", "barranco", "montana", "montanas", "roque", "casa", "iglesia",
    "centro", "ruta", "sendero", "camino", "pueblo", "ciudad", "isla", "norte",
    "sur", "este", "oeste", "zona", "para", "como", "desde", "hasta", "donde",
}


def normalize_text(text: str) -> str:
    """Minúsculas y sin acentos, para comparar nombres de lugar de forma laxa."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def place_tokens(caption: str) -> list[str]:
    """Tokens distintivos del pie de foto (sin genéricos ni palabras cortas)."""
    return [
        tok
        for tok in re.findall(r"\w+", normalize_text(caption))
        if len(tok) > 3 and tok not in PLACE_STOPWORDS
    ]


def plan_inline_images(text: str, images: list[dict]) -> list[tuple[str, object]]:
    """Planifica cómo intercalar las fotos en el texto.

    Devuelve una lista de segmentos ``("text", linea)`` y ``("images", [fotos])``:
    cada foto se coloca tras la primera línea que menciona su lugar y las no
    mencionadas se agrupan al final.
    """
    # [foto, tokens, ya_colocada]
    pending = [[img, place_tokens(img.get("caption", "")), False] for img in images]
    segments: list[tuple[str, object]] = []

    for line in text.split("\n"):
        segments.append(("text", line))
        normalized = normalize_text(line)
        matched = [
            item
            for item in pending
            if not item[2] and item[1] and any(tok in normalized for tok in item[1])
        ]
        if matched:
            segments.append(("images", [item[0] for item in matched]))
            for item in matched:
                item[2] = True

    remaining = [item[0] for item in pending if not item[2]]
    if remaining:
        segments.append(("images", remaining))
    return segments
