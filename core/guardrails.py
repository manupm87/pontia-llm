"""Guardarraíles de entrada y salida para el asistente (defensa en profundidad).

Inspirado en la sesión 05 del máster (input/output guardrails). Añade una capa
de validación independiente del prompt:

- Entrada: detección de intentos de manipulación ("prompt injection"/jailbreak)
  mediante reglas rápidas, y clasificación opcional de tema (dentro/fuera del
  ámbito turístico de Tenerife) mediante un clasificador LLM inyectable.
- Salida: juez opcional de fidelidad ("grounding") que comprueba que la respuesta
  se apoya en el contexto recuperado.

La lógica de reglas es pura y testeable; las comprobaciones con LLM se inyectan
para poder probarlas con dobles y para no encarecer cada turno si no se usan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .text import normalize_text

# Mensajes de rechazo (en español, de cara al usuario).
REFUSAL_JAILBREAK = (
    "No puedo ayudarte con eso. Estoy aquí para ayudarte a planificar tu viaje "
    "a Tenerife: playas, rutas, gastronomía, cultura, el tiempo o el estado del mar."
)
REFUSAL_OFF_TOPIC = (
    "Solo puedo ayudarte con turismo en Tenerife (lugares, playas, rutas, "
    "gastronomía, cultura, el tiempo y el mar). ¿Sobre qué de la isla te ayudo?"
)

# Patrones de manipulación de instrucciones más habituales (texto normalizado).
# Se acotan con ``.{0,40}`` para exigir cercanía entre los dos términos y evitar
# falsos positivos en preguntas turísticas legítimas (p. ej. "sistema de senderos",
# "guía completa"). Se usa ``.`` (no ``[^.]``) para que un punto intermedio no
# rompa el emparejado: "ignora. todas las instrucciones" debe detectarse igual. El
# cuantificador está acotado ({0,40}), de modo que no introduce riesgo de ReDoS.
_INJECTION_PATTERNS = [
    re.compile(p)
    for p in (
        # Anular/olvidar instrucciones, reglas o el prompt (ES/EN).
        r"\b(ignora\w*|olvida\w*|olvidate|ignore|forget|disregard)\b.{0,40}"
        r"\b(instruccion\w*|reglas?|normas?|directrices|prompt|instructions?|rules?)\b",
        # Pedir el prompt/instrucciones del sistema.
        r"system\s+prompt|prompt\s+del?\s+sistema",
        r"\b(muestrame|ensename|revela|reveal|dime|imprime|repite|repeat)\b.{0,40}"
        r"\b(tu\s+)?(prompt|instruccion\w*|instructions?)\b",
        # Modos sin restricciones / jailbreak.
        r"developer\s+mode|modo\s+desarrollador|jailbreak",
        r"\bsin\s+(censura|moderacion)\b",
        # Volcado literal/textual del documento.
        r"\b(imprime|copia|reproduce|transcribe|muestrame|dame|dump)\b.{0,40}"
        r"\b(literal\w*|integr\w*|textual\w*|verbatim|tal cual|palabra por palabra)\b",
    )
]


@dataclass(frozen=True)
class GuardVerdict:
    """Veredicto de un guardarraíl."""

    allowed: bool
    category: str = "ok"  # "ok" | "jailbreak" | "off_topic" | "ungrounded"
    reason: str = ""


def detect_injection(message: str) -> bool:
    """Indica si el mensaje parece un intento de manipular las instrucciones."""
    norm = normalize_text(message)
    return any(pattern.search(norm) for pattern in _INJECTION_PATTERNS)


class Guardrails:
    """Guardarraíles componibles para entrada y salida del asistente.

    ``topic_classifier(message) -> bool`` decide si la consulta está dentro del
    ámbito turístico; ``grounding_judge(answer, context) -> bool`` decide si la
    respuesta se apoya en el contexto. Ambos son opcionales: sin ellos, solo se
    aplica la detección de manipulación por reglas.
    """

    def __init__(
        self,
        *,
        topic_classifier: Callable[[str], bool] | None = None,
        grounding_judge: Callable[[str, str], bool] | None = None,
    ) -> None:
        self._topic_classifier = topic_classifier
        self._grounding_judge = grounding_judge

    def check_input(self, message: str) -> GuardVerdict:
        """Valida el mensaje del usuario antes de gastar en RAG y generación."""
        if detect_injection(message):
            return GuardVerdict(False, "jailbreak", "Posible inyección de instrucciones.")
        if self._topic_classifier is not None:
            try:
                if not self._topic_classifier(message):
                    return GuardVerdict(False, "off_topic", "Fuera del ámbito turístico.")
            except Exception:  # noqa: BLE001 - fail-open: ante fallo, no bloquear
                pass
        return GuardVerdict(True, "ok")

    def check_output(self, answer: str, context: str) -> GuardVerdict:
        """Valida la respuesta generada contra el contexto recuperado (fidelidad)."""
        if self._grounding_judge is None or not context:
            return GuardVerdict(True, "ok")
        try:
            grounded = self._grounding_judge(answer, context)
        except Exception:  # noqa: BLE001 - fail-open
            return GuardVerdict(True, "ok")
        if grounded:
            return GuardVerdict(True, "ok")
        return GuardVerdict(False, "ungrounded", "La respuesta no se apoya en la guía.")

    @staticmethod
    def refusal_for(verdict: GuardVerdict) -> str:
        """Mensaje de rechazo adecuado para un veredicto bloqueado."""
        if verdict.category == "off_topic":
            return REFUSAL_OFF_TOPIC
        return REFUSAL_JAILBREAK


def build_llm_guardrails(llm) -> Guardrails:
    """Construye guardarraíles con clasificador de tema y juez de fidelidad LLM.

    Usa ``with_structured_output`` sobre el modelo dado. Se mantiene fuera de la
    ruta de tests (requiere un LLM real) y se usa en la app y en la evaluación.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import BaseModel, Field

    class _TopicCheck(BaseModel):
        on_topic: bool = Field(
            description=(
                "True si la consulta trata sobre turismo, viajes, lugares, "
                "playas, rutas, gastronomía, cultura, el tiempo o el mar de "
                "Tenerife (incluye saludos y preguntas de seguimiento)."
            )
        )

    class _GroundingCheck(BaseModel):
        grounded: bool = Field(
            description="True si TODA la información de la respuesta está respaldada por el contexto."
        )

    topic_model = llm.with_structured_output(_TopicCheck)
    grounding_model = llm.with_structured_output(_GroundingCheck)

    def classify_topic(message: str) -> bool:
        messages = [
            SystemMessage(content="Clasifica si la consulta está dentro del ámbito turístico de Tenerife."),
            HumanMessage(content=message),
        ]
        return topic_model.invoke(messages).on_topic

    def judge_grounding(answer: str, context: str) -> bool:
        messages = [
            SystemMessage(content="Eres un evaluador estricto de fidelidad al contexto."),
            HumanMessage(
                content=f"CONTEXTO:\n{context}\n\nRESPUESTA:\n{answer}\n\n"
                "¿Toda la respuesta se apoya únicamente en el contexto?"
            ),
        ]
        return grounding_model.invoke(messages).grounded

    return Guardrails(topic_classifier=classify_topic, grounding_judge=judge_grounding)
