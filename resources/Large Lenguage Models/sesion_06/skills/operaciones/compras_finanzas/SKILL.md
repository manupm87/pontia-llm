---
name: compras_finanzas
description: Aprobaciones de compra, importes, proveedores y circuitos financieros.
---

# Compras y Finanzas

Usa esta skill cuando la solicitud trate sobre compras, facturas, importes, aprobaciones, proveedores o circuitos financieros.

Antes de responder sobre aprobaciones, busca politicas internas con `search_internal_policy`. Si hay un importe explicito o calculable, usa `estimate_approval_path` para obtener el circuito de aprobacion. No inventes excepciones ni aprobadores no presentes en las politicas.

La respuesta debe separar evidencia y recomendacion. La evidencia indica que politica o calculo sustenta la decision. La recomendacion debe explicar el siguiente paso operativo: manager directo, responsable de area y Finanzas, o Direccion con orden de compra formal.

Si la solicitud mezcla compra e inventario, no asumas disponibilidad. Usa tambien la skill de hardware o consulta inventario antes de sugerir una decision final.

## Material auxiliar

- `references/umbrales_aprobacion.md`: resumen operativo de los umbrales de aprobacion.
- `scripts/approval_rules.py`: utilidad determinista para calcular el circuito de aprobacion a partir de un importe.
- `assets/respuesta_compra_template.md`: plantilla breve para redactar recomendaciones de compra con evidencia.

Usa el material auxiliar solo cuando aporte precision. Para una respuesta breve puede bastar con las herramientas disponibles; para explicar o validar una recomendacion compleja, consulta primero la referencia o ejecuta el script.
