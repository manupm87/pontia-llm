---
name: hardware_it
description: Inventario de hardware, stock, plazos de entrega y asignacion tecnica.
---

# Hardware e IT

Usa esta skill cuando la solicitud incluya hardware corporativo, stock, disponibilidad, plazos de entrega o asignacion tecnica.

Para productos internos, consulta `check_inventory` antes de responder. Indica unidades disponibles, unidades solicitadas, unidades faltantes, almacen, plazo y coste estimado cuando existan datos. Si el stock no cubre la solicitud completa, propón una entrega parcial o una compra adicional, pero no confirmes disponibilidad completa.

Cuando la solicitud este relacionada con nuevas incorporaciones o priorizacion de equipos, consulta tambien politicas internas si necesitas justificar criterios de asignacion.

Si la solicitud incluye aprobaciones o importes, coordina tu respuesta con la skill de compras y Finanzas. IT puede informar disponibilidad y coste, pero no debe decidir el circuito financiero por si solo.

## Material auxiliar

- `references/catalogo_hardware.md`: catalogo de productos internos usados en los ejemplos de la sesion.
- `scripts/inventory_summary.py`: script pequeno para calcular stock faltante y coste estimado a partir de unidades disponibles, unidades solicitadas y precio unitario.

Consulta estos recursos cuando necesites explicar de forma reproducible el calculo de disponibilidad o mostrar al alumno como encapsular logica auxiliar dentro de una skill.
