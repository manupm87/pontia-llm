from __future__ import annotations

from datetime import datetime
from typing import Any

from core.data_sources import BusinessDataset, InventoryCatalog, KnowledgeBase
from core.models import ExpenseDraft, ToolSpec


TOOL_GROUPS = {
    "Conocimiento interno": ("search_company_knowledge",),
    "Inventario": ("check_inventory", "list_inventory"),
    "Métricas comerciales": ("summarize_revenue_by_region", "get_metric_trend", "get_business_overview"),
    "Borradores de gasto": ("create_expense_report_draft",),
}


class OpsToolbox:
    def __init__(
        self,
        *,
        knowledge_base: KnowledgeBase,
        inventory: InventoryCatalog,
        business: BusinessDataset,
        draft_store: dict[str, ExpenseDraft],
    ) -> None:
        self._knowledge_base = knowledge_base
        self._inventory = inventory
        self._business = business
        self._draft_store = draft_store

    def specs(self, enabled_groups: set[str]) -> list[ToolSpec]:
        enabled_names = {
            tool_name
            for group in enabled_groups
            for tool_name in TOOL_GROUPS.get(group, ())
        }
        return [spec for spec in self._all_specs() if spec.name in enabled_names]

    def search_company_knowledge(self, query: str, k: int) -> dict[str, Any]:
        return self._knowledge_base.search(query, k=k)

    def check_inventory(self, product_id: str) -> dict[str, Any]:
        return self._inventory.find(product_id)

    def list_inventory(self, low_stock_only: bool) -> list[dict[str, Any]]:
        return self._inventory.list_products(low_stock_only=low_stock_only)

    def summarize_revenue_by_region(self) -> list[dict[str, Any]]:
        return self._business.summarize_by_region()

    def get_metric_trend(self, metric: str, region: str) -> list[dict[str, Any]]:
        selected_region = None if region == "all" else region
        return self._business.metric_trend(metric, region=selected_region)

    def get_business_overview(self) -> dict[str, Any]:
        return self._business.overview()

    def create_expense_report_draft(
        self,
        vendor: str,
        amount_eur: float,
        concept: str,
        cost_center: str,
    ) -> dict[str, Any]:
        if amount_eur <= 0:
            raise ValueError("El importe debe ser mayor que cero.")

        draft_id = f"expense_draft_{len(self._draft_store) + 1:04d}"
        draft = ExpenseDraft(
            draft_id=draft_id,
            vendor=vendor,
            amount_eur=round(float(amount_eur), 2),
            concept=concept,
            cost_center=cost_center,
            requires_prior_approval=amount_eur > 500,
            status="requires_human_confirmation",
            created_at=datetime.now().isoformat(timespec="minutes"),
        )
        self._draft_store[draft_id] = draft
        payload = draft.as_dict()
        payload["message"] = "Borrador creado. No se ha enviado ni aprobado ningún gasto."
        return payload

    def _all_specs(self) -> list[ToolSpec]:
        product_ids = self._inventory.product_ids
        regions = ["all", "AMER", "EMEA", "APAC"]
        metrics = ["revenue", "cost", "active_users", "new_signups", "churn_rate", "nps"]

        return [
            ToolSpec(
                schema={
                    "type": "function",
                    "name": "search_company_knowledge",
                    "description": (
                        "Busca fragmentos relevantes en la documentación interna. "
                        "Úsala para políticas, aprobaciones, gastos, compras, RRHH, IT, seguridad, incidencias, SLA o continuidad."
                    ),
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Búsqueda semántica que representa la necesidad informativa.",
                            },
                            "k": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 8,
                                "description": "Número de fragmentos a recuperar.",
                            },
                        },
                        "required": ["query", "k"],
                        "additionalProperties": False,
                    },
                },
                function=self.search_company_knowledge,
            ),
            ToolSpec(
                schema={
                    "type": "function",
                    "name": "check_inventory",
                    "description": "Consulta stock, precio, almacén y plazo estimado de un producto interno.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_id": {
                                "type": "string",
                                "enum": product_ids,
                                "description": "Identificador interno del producto.",
                            }
                        },
                        "required": ["product_id"],
                        "additionalProperties": False,
                    },
                },
                function=self.check_inventory,
            ),
            ToolSpec(
                schema={
                    "type": "function",
                    "name": "list_inventory",
                    "description": "Lista productos de inventario. Úsala para consultas sobre catálogo o productos con stock bajo.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "low_stock_only": {
                                "type": "boolean",
                                "description": "True para devolver solo productos con 5 unidades o menos.",
                            }
                        },
                        "required": ["low_stock_only"],
                        "additionalProperties": False,
                    },
                },
                function=self.list_inventory,
            ),
            ToolSpec(
                schema={
                    "type": "function",
                    "name": "summarize_revenue_by_region",
                    "description": "Resume ingresos, costes, margen, usuarios, altas, churn y NPS por región.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
                function=self.summarize_revenue_by_region,
            ),
            ToolSpec(
                schema={
                    "type": "function",
                    "name": "get_metric_trend",
                    "description": "Devuelve una serie mensual de una métrica comercial por región o agregada.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "metric": {
                                "type": "string",
                                "enum": metrics,
                                "description": "Métrica a consultar.",
                            },
                            "region": {
                                "type": "string",
                                "enum": regions,
                                "description": "Región concreta o all para agregado global.",
                            },
                        },
                        "required": ["metric", "region"],
                        "additionalProperties": False,
                    },
                },
                function=self.get_metric_trend,
            ),
            ToolSpec(
                schema={
                    "type": "function",
                    "name": "get_business_overview",
                    "description": "Devuelve KPIs globales disponibles en el dataset comercial.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
                function=self.get_business_overview,
            ),
            ToolSpec(
                schema={
                    "type": "function",
                    "name": "create_expense_report_draft",
                    "description": (
                        "Crea un borrador de reporte de gasto pendiente de confirmación humana. "
                        "No envía, no aprueba y no registra definitivamente el gasto."
                    ),
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vendor": {"type": "string", "description": "Proveedor o comercio."},
                            "amount_eur": {"type": "number", "description": "Importe en euros."},
                            "concept": {"type": "string", "description": "Motivo del gasto."},
                            "cost_center": {
                                "type": "string",
                                "description": "Centro de coste. Usa Operaciones si el usuario no especifica otro.",
                            },
                        },
                        "required": ["vendor", "amount_eur", "concept", "cost_center"],
                        "additionalProperties": False,
                    },
                },
                function=self.create_expense_report_draft,
            ),
        ]
