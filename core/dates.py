"""Resolución de fechas relativas a formato ISO (``YYYY-MM-DD``).

El LLM razona en lenguaje natural ("mañana", "este finde", "el miércoles"), pero
las herramientas meteorológicas y marinas necesitan una fecha exacta. Este módulo
traduce esas expresiones a una fecha ISO de forma determinista, evitando que el
modelo "invente" fechas. Es lógica pura (sin red ni dependencias pesadas), por lo
que se puede probar inyectando ``today``.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from .text import normalize_text

# Horizonte máximo (días) admitido para un desplazamiento relativo: coincide con
# el alcance típico de la previsión de Open-Meteo.
_MAX_OFFSET_DAYS = 16

# Días de la semana en español -> índice de ``date.weekday`` (lunes = 0).
_WEEKDAYS = {
    "lunes": 0,
    "martes": 1,
    "miercoles": 2,
    "jueves": 3,
    "viernes": 4,
    "sabado": 5,
    "domingo": 6,
}

# Palabras que indican "la semana que viene" (saltan 7 días si caería en hoy).
_NEXT_WEEK_HINTS = ("proximo", "proxima", "siguiente", "que viene")

_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def resolve_date(expression: str, *, today: date | None = None) -> str:
    """Convierte una expresión de fecha (relativa o absoluta) a ISO ``YYYY-MM-DD``.

    Admite fechas absolutas (``2026-12-25``), palabras clave (``hoy``, ``mañana``,
    ``pasado mañana``, ``finde``), días de la semana (``el miércoles``,
    ``próximo domingo``) y desplazamientos (``en 3 días``, ``+5``). Lanza
    ``ValueError`` si no puede interpretar la expresión.
    """
    base = today or date.today()
    text = normalize_text(expression)
    tokens = set(re.findall(r"\w+", text))

    # Fecha absoluta ya en ISO: se valida y se devuelve tal cual.
    if _ISO_RE.match(text):
        date.fromisoformat(text)  # valida; propaga ValueError si es inválida
        return text

    # El orden importa: lo más específico primero. "pasado mañana" antes que
    # "mañana"; los días de la semana antes que "mañana" (para no confundir
    # "el sábado por la mañana" con "mañana").
    if text in ("", "today") or "hoy" in text:
        return base.isoformat()
    if "pasado manana" in text:
        return (base + timedelta(days=2)).isoformat()

    # Fin de semana -> próximo sábado, salvo que hoy ya sea sábado o domingo: en
    # ese caso "este finde" se refiere al fin de semana en curso (devuelve hoy).
    if "finde" in text or "fin de semana" in text:
        if base.weekday() in (_WEEKDAYS["sabado"], _WEEKDAYS["domingo"]):
            return base.isoformat()
        return _next_weekday(base, _WEEKDAYS["sabado"], force_next_week=False).isoformat()

    # Día de la semana con nombre ("el miércoles", "el lunes que viene"). Se exige
    # que el nombre aparezca como palabra completa (token), no como subcadena, para
    # no confundir frases como "mañana es martes" con un cambio de día objetivo.
    for name, index in _WEEKDAYS.items():
        if name in tokens:
            force = any(hint in text for hint in _NEXT_WEEK_HINTS)
            return _next_weekday(base, index, force_next_week=force).isoformat()

    if "manana" in text:
        return (base + timedelta(days=1)).isoformat()

    # "una/N semana(s)" -> múltiplos de 7 días.
    if "semana" in text:
        match = re.search(r"\d+", text)
        weeks = int(match.group()) if match else 1
        offset = weeks * 7
        if 0 <= offset <= _MAX_OFFSET_DAYS:
            return (base + timedelta(days=offset)).isoformat()

    # Desplazamiento numérico ("en 3 días", "dentro de 4 días", "3", "+5").
    if "dia" in text or re.fullmatch(r"\+?\d+", text):
        match = re.search(r"\d+", text)
        if match:
            offset = int(match.group())
            if 0 <= offset <= _MAX_OFFSET_DAYS:
                return (base + timedelta(days=offset)).isoformat()

    raise ValueError(
        f"No se pudo interpretar la fecha '{expression}'. Usa una fecha en "
        "formato YYYY-MM-DD o una expresión relativa (como 'mañana' o 'este "
        f"finde') dentro de los próximos {_MAX_OFFSET_DAYS} días."
    )


def _next_weekday(base: date, target: int, *, force_next_week: bool) -> date:
    """Devuelve la próxima fecha con el día de la semana ``target``.

    Si ``base`` ya cae en ese día, devuelve ``base`` salvo que ``force_next_week``
    pida saltar a la semana siguiente.
    """
    delta = (target - base.weekday()) % 7
    if force_next_week:
        # "que viene" / "próximo": el de la semana siguiente, no el más cercano.
        delta += 7
    return base + timedelta(days=delta)
