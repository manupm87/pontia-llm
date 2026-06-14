from __future__ import annotations

import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP


SESSION_DIR = Path(__file__).resolve().parents[1]
if str(SESSION_DIR) not in sys.path:
    sys.path.insert(0, str(SESSION_DIR))

from operations_support import get_inventory_status, search_policy_documents  # noqa: E402


mcp = FastMCP(
    "operaciones-internas",
    instructions=(
        "Servidor MCP de ejemplo para la sesión 6. Expone políticas internas "
        "e inventario ficticio de operaciones."
    ),
)


@mcp.tool()
def search_policy(query: str) -> dict[str, list[dict[str, object]]]:
    """Busca políticas internas relevantes para una consulta."""
    return {"results": search_policy_documents(query, k=3)}


@mcp.tool()
def inventory_status(product_id: str, requested_units: int = 1) -> dict[str, object]:
    """Consulta stock, coste y disponibilidad de un producto de hardware."""
    return get_inventory_status(product_id, requested_units=requested_units)


@mcp.resource("policy://{area}")
def policy_by_area(area: str) -> str:
    """Devuelve políticas de un área concreta como recurso MCP."""
    matches = [
        policy
        for policy in search_policy_documents(area, k=5)
        if policy["area"].lower() == area.lower()
    ]
    if not matches:
        return f"No hay políticas para el área {area}."
    return "\n\n".join(f"{item['title']}\n{item['text']}" for item in matches)


if __name__ == "__main__":
    mcp.run(transport="stdio")
