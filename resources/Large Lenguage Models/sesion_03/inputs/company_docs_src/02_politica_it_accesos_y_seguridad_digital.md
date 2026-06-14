---
title: "Política de IT, accesos y seguridad digital"
subtitle: "NovaCorp Iberia · IT Operations"
author: "Dirección de Tecnología"
date: "Versión 4.1 · Enero 2026"
lang: es
---

# Control del documento

| Campo | Valor |
|---|---|
| Propietario | IT Operations |
| Ámbito | Empleados, colaboradores y proveedores con acceso corporativo |
| Clasificación | Uso interno |
| Sistemas cubiertos | Correo, VPN, portal, dispositivos, SaaS corporativo |
| Canal de soporte | Portal de empleados > Soporte IT |

# 1. Principios generales

Los accesos corporativos se conceden con el criterio de mínimo privilegio. Cada usuario debe tener únicamente los permisos necesarios para desempeñar su función. Las cuentas son personales e intransferibles; queda prohibido compartir contraseñas, códigos MFA o sesiones abiertas.

IT puede suspender temporalmente un acceso si detecta indicios de compromiso, uso anómalo o incumplimiento de esta política.

# 2. Acceso al portal de empleados desde casa

El portal de empleados está disponible en `https://portal.novacorp.example`. Para acceder desde fuera de una oficina de NovaCorp se requiere:

1. conexión a internet estable y no compartida públicamente;
2. VPN corporativa activa;
3. usuario corporativo en formato `nombre.apellido@novacorp.example`;
4. contraseña vigente;
5. autenticación multifactor aprobada desde la aplicación autorizada.

Si la VPN no conecta, se debe comprobar primero que el portátil tiene conexión, que la hora del sistema es correcta y que no hay otra VPN activa. Si el problema continúa, se abre un ticket en la categoría **IT > Conectividad > VPN**.

# 3. Contraseñas y recuperación de cuenta

## 3.1 Reglas de contraseña

La contraseña debe tener al menos catorce caracteres y combinar letras, números y símbolos. No se permite reutilizar contraseñas de servicios personales. El sistema bloquea las diez últimas contraseñas utilizadas.

## 3.2 Olvido de contraseña del correo corporativo

Si una persona olvida la contraseña del correo corporativo debe usar el autoservicio de recuperación:

1. entrar en `https://login.novacorp.example`;
2. seleccionar **No puedo acceder a mi cuenta**;
3. introducir el correo corporativo;
4. validar identidad mediante MFA o correo alternativo registrado;
5. definir una nueva contraseña;
6. cerrar sesiones abiertas y volver a iniciar sesión.

Si no tiene acceso al segundo factor, debe abrir un ticket desde el portal externo de soporte de identidad o pedir al manager que solicite una verificación manual. IT nunca pedirá la contraseña por chat, correo o llamada.

# 4. Autenticación multifactor

MFA es obligatorio para correo, VPN, portal de empleados, repositorio documental, CRM, consola cloud y herramientas financieras. Los métodos permitidos son aplicación autenticadora corporativa y llave física FIDO2 para perfiles de alto privilegio.

No se permite aprobar notificaciones MFA no solicitadas. Si aparece una solicitud que el usuario no reconoce, debe rechazarla y reportar incidente de seguridad.

# 5. Solicitud de accesos

Los accesos se solicitan desde el catálogo de servicios. El flujo estándar es:

1. empleado selecciona aplicación y rol;
2. manager aprueba necesidad funcional;
3. propietario de la aplicación valida el nivel de permiso;
4. IT provisiona el acceso;
5. el sistema registra fecha, aprobadores y caducidad si aplica.

Los accesos privilegiados caducan automáticamente a los noventa días salvo renovación justificada.

# 6. Dispositivos corporativos

Todo portátil corporativo debe tener cifrado de disco, agente EDR, bloqueo automático a los cinco minutos y actualizaciones activas. No se permite instalar software sin licencia o herramientas que eviten controles de seguridad.

La pérdida o robo de un dispositivo debe reportarse en menos de una hora desde su detección. IT ejecutará borrado remoto si existe riesgo para datos corporativos.

# 7. Correo y phishing

El correo corporativo debe utilizarse solo para actividad profesional. Los mensajes sospechosos se reportan con el botón **Reportar phishing** del cliente de correo. No se deben reenviar adjuntos sospechosos a compañeros.

Se considera señal de riesgo: urgencia inusual, enlaces acortados, solicitud de credenciales, archivos comprimidos inesperados, cambios de cuenta bancaria o mensajes que simulan provenir de dirección ejecutiva.

# 8. Niveles de servicio

| Tipo de incidencia | Ejemplo | Prioridad | Objetivo de primera respuesta |
|---|---|---|---|
| Bloqueo total | no puede iniciar sesión ni acceder al correo | Alta | 2 horas laborables |
| Degradación importante | VPN intermitente, correo lento | Media | 4 horas laborables |
| Petición estándar | alta de herramienta aprobada | Normal | 2 días laborables |
| Consulta | duda de configuración | Baja | 3 días laborables |

# 9. Criterios de escalado

Se escala a Seguridad si hay indicios de cuenta comprometida, phishing con credenciales, pérdida de equipo o acceso no autorizado. Se escala al propietario de aplicación si el problema afecta a permisos funcionales. Se escala a Personas si el acceso depende de cambios contractuales o de rol.
