"""Tests de ``core.dates.resolve_date`` (resolución de fechas relativas).

Se inyecta ``today`` para que las aserciones sean deterministas sin depender
del reloj real. El 2026-06-14 es domingo (sirve de ancla en varios casos).
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from core.dates import resolve_date

# Domingo 14 de junio de 2026 como "hoy" de referencia.
SUNDAY = date(2026, 6, 14)


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("hoy", SUNDAY),
        ("", SUNDAY),
        ("today", SUNDAY),
        ("mañana", SUNDAY + timedelta(days=1)),
        ("manana", SUNDAY + timedelta(days=1)),
        ("  Mañana  ", SUNDAY + timedelta(days=1)),
        ("pasado mañana", SUNDAY + timedelta(days=2)),
        ("pasado manana", SUNDAY + timedelta(days=2)),
    ],
)
def test_relative_keywords(expression: str, expected: date) -> None:
    assert resolve_date(expression, today=SUNDAY) == expected.isoformat()


@pytest.mark.parametrize(
    "expression, days",
    [("3", 3), ("+5", 5), ("en 2 dias", 2), ("dentro de 4 días", 4)],
)
def test_day_offsets(expression: str, days: int) -> None:
    expected = SUNDAY + timedelta(days=days)
    assert resolve_date(expression, today=SUNDAY) == expected.isoformat()


def test_weekday_same_day_returns_today() -> None:
    # Hoy es domingo: "domingo" debe resolver a hoy mismo.
    assert resolve_date("domingo", today=SUNDAY) == SUNDAY.isoformat()


def test_weekday_next_occurrence() -> None:
    # Desde el domingo, el próximo "miércoles" es el 17.
    assert resolve_date("miércoles", today=SUNDAY) == date(2026, 6, 17).isoformat()


def test_weekday_with_proximo_skips_a_week() -> None:
    # "próximo domingo" desde el domingo salta a la semana siguiente.
    assert resolve_date("próximo domingo", today=SUNDAY) == date(2026, 6, 21).isoformat()


def test_weekday_que_viene_skips_to_next_week() -> None:
    # "el lunes que viene": no el lunes más cercano (15) sino el de la semana
    # siguiente (22). Cubre el bug de la pista "que viene" en días != hoy.
    assert resolve_date("el lunes que viene", today=SUNDAY) == date(2026, 6, 22).isoformat()


@pytest.mark.parametrize("expression", ["para mañana", "mañana por la mañana"])
def test_manana_with_surrounding_words(expression: str) -> None:
    assert resolve_date(expression, today=SUNDAY) == (SUNDAY + timedelta(days=1)).isoformat()


def test_weekday_takes_precedence_over_morning_word() -> None:
    # "el sábado por la mañana" es sábado (20), no 'mañana' (lunes 15).
    assert resolve_date("el sábado por la mañana", today=SUNDAY) == date(2026, 6, 20).isoformat()


def test_hoy_with_surrounding_words() -> None:
    assert resolve_date("hoy por la mañana", today=SUNDAY) == SUNDAY.isoformat()


@pytest.mark.parametrize(
    "expression, days",
    [("una semana", 7), ("dentro de una semana", 7), ("dentro de 2 semanas", 14)],
)
def test_week_expressions(expression: str, days: int) -> None:
    expected = SUNDAY + timedelta(days=days)
    assert resolve_date(expression, today=SUNDAY) == expected.isoformat()


@pytest.mark.parametrize("expression", ["finde", "fin de semana", "este finde"])
def test_weekend_on_weekend_resolves_to_today(expression: str) -> None:
    # Hoy es domingo (fin de semana en curso): "este finde" debe ser hoy mismo,
    # no el próximo sábado. Cubre el bug que saltaba el fin de semana actual.
    assert resolve_date(expression, today=SUNDAY) == SUNDAY.isoformat()


@pytest.mark.parametrize("expression", ["finde", "fin de semana", "este finde"])
def test_weekend_midweek_resolves_to_next_saturday(expression: str) -> None:
    # Entre semana (miércoles 17), "este finde" es el próximo sábado (20).
    wednesday = date(2026, 6, 17)
    assert resolve_date(expression, today=wednesday) == date(2026, 6, 20).isoformat()


@pytest.mark.parametrize("expression", ["el martesx", "lunescheck", "sabadomingo"])
def test_weekday_substring_does_not_misfire(expression: str) -> None:
    # Un nombre de día incrustado en una palabra mayor ("martesx") no debe contar
    # como ese día: solo se empareja como token (palabra completa). Antes, el
    # emparejado por subcadena lo confundía; ahora es ininterpretable -> ValueError.
    with pytest.raises(ValueError):
        resolve_date(expression, today=SUNDAY)


def test_absolute_iso_date_passes_through() -> None:
    assert resolve_date("2026-12-25", today=SUNDAY) == "2026-12-25"


@pytest.mark.parametrize("expression", ["cualquier cosa", "el año que viene", "32"])
def test_unparseable_raises_value_error(expression: str) -> None:
    with pytest.raises(ValueError):
        resolve_date(expression, today=SUNDAY)


def test_uses_real_today_by_default() -> None:
    # Sin inyectar ``today`` usa la fecha real del sistema.
    assert resolve_date("hoy") == date.today().isoformat()
