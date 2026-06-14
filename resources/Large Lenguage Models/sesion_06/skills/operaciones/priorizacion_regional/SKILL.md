---
name: priorizacion_regional
description: Priorizacion por region usando revenue, margen, pipeline y conversion.
---

# Priorizacion regional

Usa esta skill cuando la solicitud mencione EMEA, LATAM, NA, pipeline, revenue, margen, conversion o impacto comercial regional.

Consulta `summarize_region_metrics` antes de justificar prioridad regional. Compara regiones solo con los datos observados. No inventes objetivos comerciales, cuotas ni forecasts fuera del dataset disponible.

La respuesta debe explicar si la region parece prioritaria y por que. Si la evidencia regional no basta para justificar una compra o cambio operativo, dilo explicitamente.

Cuando una solicitud regional implique gasto, esta skill solo aporta el argumento de prioridad. La aprobacion presupuestaria debe venir de la skill de compras y Finanzas.

## Material auxiliar

- `references/criterios_priorizacion.md`: criterios cualitativos para interpretar revenue, margen, pipeline y conversion.

Usa la referencia cuando tengas que justificar por que una metrica regional apoya o no apoya la priorizacion. No sustituyas los datos observados por la referencia; la referencia solo explica como leerlos.
