from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from agents import function_tool


@dataclass(frozen=True)
class PolicyDocument:
    source: str
    area: str
    title: str
    text: str


@dataclass(frozen=True)
class InventoryRecord:
    product_id: str
    name: str
    available_units: int
    unit_price_eur: float
    warehouse: str
    lead_time_days: int
    supplier: str


@dataclass(frozen=True)
class RegionMetric:
    region: str
    revenue_eur: float
    operating_cost_eur: float
    pipeline_eur: float
    conversion_rate: float


@dataclass(frozen=True)
class SupportTicket:
    ticket_id: str
    area: str
    summary: str
    priority: Literal["low", "medium", "high"]
    status: Literal["open", "waiting", "resolved"]


POLICY_DOCUMENTS = [
    PolicyDocument(
        source="finanzas_compras.md",
        area="Finanzas",
        title="Compras y aprobaciones de gasto",
        text=(
            "Las compras inferiores a 500 EUR pueden aprobarse por el manager directo. "
            "Entre 500 y 2.000 EUR se requiere aprobación del responsable de área y validación de Finanzas. "
            "Por encima de 2.000 EUR se requiere aprobación de Dirección y orden de compra formal. "
            "Toda compra de hardware debe incluir centro de coste, proveedor recomendado y justificación de necesidad."
        ),
    ),
    PolicyDocument(
        source="it_hardware.md",
        area="IT",
        title="Asignación de hardware corporativo",
        text=(
            "El hardware corporativo se asigna priorizando nuevas incorporaciones, equipos comerciales y soporte crítico. "
            "Monitores, docks y portátiles deben revisarse contra inventario antes de confirmar plazos. "
            "Si el stock disponible no cubre la solicitud completa, IT debe proponer entrega parcial o compra adicional."
        ),
    ),
    PolicyDocument(
        source="rrhh_onboarding.md",
        area="RRHH",
        title="Onboarding y necesidades de puesto",
        text=(
            "RRHH coordina onboarding con manager, IT y Finanzas. "
            "Cada nueva incorporación debe tener fecha de inicio, rol, región, manager y necesidades de hardware. "
            "Las solicitudes incompletas deben devolverse para completar datos antes de iniciar compras o altas de acceso."
        ),
    ),
    PolicyDocument(
        source="seguridad_accesos.md",
        area="Seguridad",
        title="Accesos y herramientas con datos sensibles",
        text=(
            "Toda herramienta SaaS con acceso a datos de cliente requiere revisión de Seguridad antes de contratación. "
            "Las altas con permisos administrativos necesitan aprobación explícita del responsable de Seguridad. "
            "Las incidencias de credenciales comprometidas tienen prioridad alta y deben escalarse inmediatamente."
        ),
    ),
    PolicyDocument(
        source="operaciones_regionales.md",
        area="Operaciones",
        title="Priorización regional",
        text=(
            "Las solicitudes regionales se priorizan según impacto en revenue, margen, pipeline y urgencia operativa. "
            "EMEA tiene prioridad cuando la solicitud desbloquea oportunidades comerciales activas. "
            "LATAM requiere control estricto de coste si el margen operativo cae por debajo del 35%."
        ),
    ),
]


INVENTORY = [
    InventoryRecord("monitor-27", "Monitor 27 pulgadas", 8, 219.0, "Madrid", 2, "DisplayPro"),
    InventoryRecord("laptop-pro-14", "Portátil Pro 14", 5, 1499.0, "Madrid", 5, "Northwind Hardware"),
    InventoryRecord("dock-usb-c", "Dock USB-C", 23, 139.0, "Barcelona", 3, "CableWorks"),
    InventoryRecord("security-token", "Llave de seguridad FIDO2", 40, 34.0, "Madrid", 1, "SecureKey"),
]


REGION_METRICS = [
    RegionMetric("EMEA", 1_820_000, 1_090_000, 640_000, 0.31),
    RegionMetric("LATAM", 720_000, 490_000, 210_000, 0.24),
    RegionMetric("NA", 2_450_000, 1_520_000, 780_000, 0.35),
]


SUPPORT_TICKETS = [
    SupportTicket("IT-1042", "IT", "Stock parcial de monitores para EMEA", "medium", "open"),
    SupportTicket("SEC-2088", "Seguridad", "Revisión de SaaS con datos de cliente", "high", "waiting"),
    SupportTicket("HR-0911", "RRHH", "Onboarding pendiente para dos SDR en EMEA", "medium", "open"),
]


WORD_PATTERN = re.compile(r"[a-záéíóúüñ0-9-]+", re.IGNORECASE)


def tokenize(text: str) -> set[str]:
    return {match.group(0).lower() for match in WORD_PATTERN.finditer(text)}


def to_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def search_policy_documents(query: str, *, k: int = 3) -> list[dict[str, Any]]:
    query_terms = tokenize(query)
    scored: list[tuple[int, PolicyDocument]] = []

    for document in POLICY_DOCUMENTS:
        searchable = f"{document.area} {document.title} {document.text}"
        document_terms = tokenize(searchable)
        lexical_score = len(query_terms & document_terms)
        area_boost = 2 if document.area.lower() in query.lower() else 0
        score = lexical_score + area_boost
        if score > 0:
            scored.append((score, document))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "source": document.source,
            "area": document.area,
            "title": document.title,
            "text": document.text,
            "score": score,
        }
        for score, document in scored[:k]
    ]


def get_inventory_status(product_id: str, *, requested_units: int = 1) -> dict[str, Any]:
    record = next((item for item in INVENTORY if item.product_id == product_id), None)
    if record is None:
        return {
            "product_id": product_id,
            "found": False,
            "message": "Producto no encontrado en inventario.",
        }

    total_price = requested_units * record.unit_price_eur
    return {
        **asdict(record),
        "found": True,
        "requested_units": requested_units,
        "can_fulfill": record.available_units >= requested_units,
        "missing_units": max(0, requested_units - record.available_units),
        "estimated_total_eur": round(total_price, 2),
    }


def summarize_regions() -> list[dict[str, Any]]:
    rows = []
    for metric in REGION_METRICS:
        margin = metric.revenue_eur - metric.operating_cost_eur
        rows.append(
            {
                **asdict(metric),
                "margin_eur": margin,
                "margin_rate": round(margin / metric.revenue_eur, 3),
            }
        )
    return rows


def approval_path_for_amount(amount_eur: float) -> dict[str, Any]:
    if amount_eur < 500:
        path = "manager_directo"
        description = "Puede aprobarlo el manager directo."
    elif amount_eur <= 2_000:
        path = "responsable_area_y_finanzas"
        description = "Requiere responsable de área y validación de Finanzas."
    else:
        path = "direccion_y_orden_de_compra"
        description = "Requiere Dirección y orden de compra formal."

    return {
        "amount_eur": amount_eur,
        "approval_path": path,
        "description": description,
    }


@function_tool
def search_internal_policy(query: str, k: int = 3) -> str:
    """Busca políticas internas relevantes.

    Args:
        query: Consulta en lenguaje natural sobre compras, IT, RRHH, seguridad u operaciones.
        k: Número máximo de documentos a recuperar. Usa 2 o 3 salvo que necesites más cobertura.
    """
    return to_json(search_policy_documents(query, k=k))


@function_tool
def check_inventory(product_id: str, requested_units: int = 1) -> str:
    """Consulta disponibilidad, precio, almacén y plazo de entrega de un producto.

    Args:
        product_id: Identificador interno del producto. Ejemplos: monitor-27, laptop-pro-14, dock-usb-c.
        requested_units: Número de unidades solicitadas por el usuario.
    """
    return to_json(get_inventory_status(product_id, requested_units=requested_units))


@function_tool
def summarize_region_metrics() -> str:
    """Resume revenue, coste, margen, pipeline y conversión por región."""
    return to_json(summarize_regions())


@function_tool
def estimate_approval_path(amount_eur: float) -> str:
    """Estima el circuito de aprobación necesario para un importe de compra en EUR."""
    return to_json(approval_path_for_amount(amount_eur))


__all__ = [
    "INVENTORY",
    "POLICY_DOCUMENTS",
    "REGION_METRICS",
    "SUPPORT_TICKETS",
    "approval_path_for_amount",
    "check_inventory",
    "estimate_approval_path",
    "get_inventory_status",
    "search_internal_policy",
    "search_policy_documents",
    "summarize_region_metrics",
    "summarize_regions",
    "to_json",
]
