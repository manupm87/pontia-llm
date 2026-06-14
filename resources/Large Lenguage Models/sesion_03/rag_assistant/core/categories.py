from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CategoryName = Literal["it", "rrhh", "seguridad", "general"]


@dataclass(frozen=True)
class Category:
    name: CategoryName
    display_name: str
    accent: str


CATEGORIES: tuple[Category, ...] = (
    Category("it", "IT y accesos", "#0d1828"),
    Category("rrhh", "Personas y RRHH", "#143f35"),
    Category("seguridad", "Seguridad", "#f5660b"),
    Category("general", "General", "#5b625f"),
)

ALL_LABEL = "Todas las categorías"

_IT_KEYWORDS = ("contraseña", "correo", "portal", "vpn", "credencial", "acceso", "wifi", "mail")
_RRHH_KEYWORDS = ("vacaciones", "desempeño", "rrhh", "nómina", "permiso", "baja", "ausencia")
_SECURITY_KEYWORDS = ("seguridad", "incidente", "incidencia", "robo", "alarma", "emergencia")


def categorize(text: str) -> CategoryName:
    content = text.lower()
    if any(keyword in content for keyword in _IT_KEYWORDS):
        return "it"
    if any(keyword in content for keyword in _RRHH_KEYWORDS):
        return "rrhh"
    if any(keyword in content for keyword in _SECURITY_KEYWORDS):
        return "seguridad"
    return "general"


def labels() -> list[str]:
    return [ALL_LABEL, *(category.display_name for category in CATEGORIES)]


def get_by_label(label: str) -> Category | None:
    if label == ALL_LABEL:
        return None
    for category in CATEGORIES:
        if category.display_name == label:
            return category
    return None


def get_by_name(name: str) -> Category | None:
    for category in CATEGORIES:
        if category.name == name:
            return category
    return None


def default_index() -> int:
    return 0
