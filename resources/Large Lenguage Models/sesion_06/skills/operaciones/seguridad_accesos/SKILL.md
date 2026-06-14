---
name: seguridad_accesos
description: Revision de seguridad, accesos administrativos y herramientas con datos sensibles.
---

# Seguridad y accesos

Usa esta skill cuando la solicitud mencione accesos administrativos, credenciales, herramientas SaaS, datos de cliente o datos sensibles.

Consulta politicas internas con `search_internal_policy` antes de recomendar una accion. Si hay credenciales comprometidas, trata la solicitud como riesgo alto y recomienda escalado inmediato. Si se trata de contratar o revisar una herramienta SaaS con datos de cliente, exige revision de Seguridad antes de continuar.

No apruebes accesos ni compras sensibles sin evidencia de autorizacion explicita.

Si la solicitud mezcla Seguridad y compras, Seguridad debe bloquear o condicionar la compra cuando falte revision de riesgo. La disponibilidad presupuestaria no sustituye la aprobacion de Seguridad.

## Material auxiliar

- `references/matriz_riesgo.md`: criterios para clasificar solicitudes de accesos, credenciales y SaaS con datos sensibles.
- `assets/checklist_revision_saas.md`: checklist reusable para revisar herramientas SaaS antes de contratarlas.

Usa estos recursos cuando la respuesta requiera justificar un bloqueo, una escalada o una revision adicional de Seguridad.
