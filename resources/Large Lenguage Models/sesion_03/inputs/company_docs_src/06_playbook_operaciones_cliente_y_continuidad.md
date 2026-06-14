---
title: "Playbook de operaciones, atención a cliente y continuidad"
subtitle: "NovaCorp Iberia · Customer Operations"
author: "Dirección de Operaciones"
date: "Versión 2.2 · Enero 2026"
lang: es
---

# Control del documento

| Campo | Valor |
|---|---|
| Propietario | Dirección de Operaciones |
| Ámbito | Atención a clientes, incidencias de servicio y continuidad |
| Clasificación | Uso interno |
| Herramienta principal | Service Desk corporativo |
| Revisión | Semestral |

# 1. Objetivo

Este playbook define cómo gestionar solicitudes de cliente, incidencias operativas, comunicaciones y continuidad de servicio. Debe utilizarse junto con los contratos específicos de cliente y los acuerdos de nivel de servicio aplicables.

# 2. Tipos de solicitud

| Tipo | Descripción | Sistema |
|---|---|---|
| Consulta | Pregunta sin impacto en servicio | Service Desk |
| Petición | Solicitud planificada de cambio o información | Service Desk |
| Incidencia | Degradación o interrupción de servicio | Service Desk |
| Problema | Causa raíz recurrente o no resuelta | Registro de problemas |
| Cambio | Modificación en servicio, configuración o proceso | Gestión de cambios |

# 3. Registro de incidencias

Toda incidencia debe registrarse con:

- cliente afectado;
- servicio o producto;
- hora de detección;
- impacto observable;
- severidad inicial;
- evidencias disponibles;
- acciones tomadas;
- propietario temporal.

No se deben gestionar incidencias únicamente por chat. El chat puede coordinar, pero el registro oficial debe estar en la herramienta.

# 4. Severidad operativa

| Severidad | Impacto | Ejemplo | Actualización |
|---|---|---|---|
| SEV1 | Servicio crítico caído para múltiples clientes | interrupción total | cada 30 minutos |
| SEV2 | Degradación importante o cliente estratégico afectado | latencia severa | cada 60 minutos |
| SEV3 | Impacto limitado con alternativa | error parcial | cada día laborable |
| SEV4 | Consulta o petición sin impacto | duda operativa | según SLA |

La severidad puede cambiar durante la investigación. Cualquier reducción de severidad debe justificarse en el ticket.

# 5. Comunicación con clientes

La comunicación debe ser clara, factual y sin atribuir causa raíz antes de confirmarla. Se recomienda estructura:

1. reconocimiento del problema;
2. impacto conocido;
3. acciones en curso;
4. próxima actualización;
5. canal de seguimiento.

No se comparten logs internos, nombres de empleados ni información de otros clientes.

# 6. Escalado interno

Se escala a Ingeniería si la incidencia requiere cambio técnico o análisis de logs de plataforma. Se escala a Producto si afecta a comportamiento funcional. Se escala a Legal o Seguridad si hay sospecha de exposición de datos, incumplimiento contractual o incidente de seguridad.

# 7. Postmortem

Toda incidencia SEV1 y SEV2 requiere postmortem. El documento debe incluir cronología, impacto, causa raíz, detección, respuesta, mitigación, acciones preventivas y propietario de cada acción.

El postmortem no busca culpables. Busca mejorar detección, resiliencia, comunicación y procesos.

# 8. Continuidad operativa

Cada área mantiene un plan de continuidad con responsables, contactos alternativos, herramientas críticas y procedimientos manuales. El plan se prueba al menos una vez al año.

Si una herramienta crítica no está disponible, el responsable de área activa el procedimiento manual documentado y comunica el estado a Operaciones.

# 9. Cierre de incidencias

Una incidencia se cierra cuando el servicio está restaurado, el cliente ha sido informado y quedan registradas acciones pendientes si aplica. Si el cliente no responde, se puede cerrar tras dos intentos de contacto en días laborables distintos.

# 10. Criterios de escalado ejecutivo

Se informa a dirección cuando existe impacto en cliente estratégico, riesgo reputacional, exposición de datos, incumplimiento de SLA crítico o incidencia SEV1 superior a dos horas.
