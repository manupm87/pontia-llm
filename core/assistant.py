"""Asistente turístico conversacional de Tenerife.

Orquesta el diálogo multiturno con Gemini: gestiona el historial, ejecuta el
bucle de *tool calling* (RAG sobre la guía, tiempo y estado del mar) y expone
tanto una respuesta completa (``chat``) como una respuesta en *streaming* real
de tokens (``stream``). El flujo se divide en dos fases públicas —``prepare``
(resuelve las herramientas) y ``stream_answer`` (emite la respuesta)— para que
la interfaz pueda mostrar en vivo qué herramientas se están usando. Mantiene un
registro de las llamadas a herramientas para observabilidad.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Iterator

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)

from .config import Settings
from . import tools as tools_module

if TYPE_CHECKING:  # Solo para anotaciones: no arrastra las dependencias del RAG.
    from .rag import TouristGuideRAG

logger = logging.getLogger("asistente_tenerife.assistant")

# Días de la semana en español para anclar la fecha en el prompt de sistema.
_WEEKDAY_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def build_system_prompt(today: date | None = None) -> str:
    """Construye el prompt de sistema en español, anclado a la fecha actual.

    Inyectar la fecha permite que el asistente resuelva expresiones relativas
    ('hoy', 'mañana', 'este finde') a través de la herramienta ``resolve_date``.
    """
    today = today or date.today()
    weekday = _WEEKDAY_ES[today.weekday()]
    return (
        "Eres un asistente turístico experto en la isla de Tenerife. Ayudas a los "
        "turistas a planificar su viaje de forma cercana y útil.\n\n"
        f"Hoy es {weekday}, {today.isoformat()}.\n\n"
        "Reglas:\n"
        "- Para CUALQUIER pregunta sobre lugares, playas, rutas, miradores, "
        "gastronomía, cultura o cosas que ver o hacer en Tenerife DEBES llamar "
        "SIEMPRE a la herramienta 'search_tourist_guide' antes de responder, "
        "también en las preguntas de seguimiento. No respondas nunca de memoria.\n"
        "- Basa tu respuesta ÚNICAMENTE en los fragmentos que devuelva la guía y "
        "cita siempre la página. No uses tu conocimiento previo ni añadas lugares o "
        "datos que no aparezcan en los fragmentos recuperados.\n"
        "- Si la guía no contiene la respuesta, dilo con claridad y no inventes "
        "información.\n"
        "- Cuando el usuario mencione una fecha relativa (hoy, mañana, este finde, "
        "el miércoles...), usa primero 'resolve_date' para obtener la fecha exacta "
        "(YYYY-MM-DD) y pásala a 'get_weather' o 'get_sea_conditions'.\n"
        "- Usa 'get_weather' para la previsión meteorológica y 'get_sea_conditions' "
        "para el estado del mar (oleaje, baño, surf) en una fecha concreta.\n"
        "- Responde siempre en español, con un tono cercano y práctico."
    )


# Prompt por defecto (anclado a la fecha de import); se mantiene por
# compatibilidad con el notebook. El asistente lo reconstruye en cada sesión.
SYSTEM_PROMPT = build_system_prompt()


@dataclass
class ToolCallRecord:
    """Registro de una llamada a herramienta para observabilidad y trazas."""

    name: str
    arguments: dict
    ok: bool
    result: str
    elapsed_s: float


@dataclass
class TurnContext:
    """Estado de un turno tras resolver las herramientas, listo para responder."""

    working: list[BaseMessage]
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    # Respuesta final ya generada durante el bucle (sin tool calls), si la hubo;
    # permite a ``chat`` reutilizarla sin volver a invocar al modelo.
    final: AIMessage | None = None


class TouristAssistant:
    """Asistente conversacional con memoria, RAG y *function calling*."""

    def __init__(
        self, settings: Settings, rag: "TouristGuideRAG", *, llm=None
    ) -> None:
        """Inicializa el LLM con herramientas y arranca el historial.

        ``llm`` permite inyectar un modelo ya construido (útil en tests); si es
        ``None`` se crea el modelo de Gemini a partir de la configuración.
        """
        # La herramienta de RAG necesita la instancia compartida del índice.
        tools_module.set_rag_instance(rag)

        self.settings = settings
        self.rag = rag
        self.llm = llm if llm is not None else self._build_llm(settings)
        self.tools = tools_module.get_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tools_by_name = {t.name: t for t in self.tools}
        self.history: list[BaseMessage] = [SystemMessage(content=build_system_prompt())]
        self.tool_log: list[ToolCallRecord] = []
        # Fuentes y fotos citadas en la última respuesta (para Streamlit).
        self.last_sources: list[dict] = []
        self.last_images: list[dict] = []

    @staticmethod
    def _build_llm(settings: Settings):
        """Crea el modelo de Gemini (import perezoso para no exigir ``langchain``)."""
        from langchain.chat_models import init_chat_model

        return init_chat_model(
            settings.generation_model,
            model_provider="google_genai",
            temperature=settings.temperature,
            top_p=settings.top_p,
            max_tokens=settings.max_output_tokens,
        )

    def prepare(self, user_message: str, max_tool_rounds: int = 5) -> TurnContext:
        """Resuelve las rondas de *tool calling* y deja el turno listo para responder.

        Añade el mensaje del usuario al historial, ejecuta las herramientas sobre
        una copia efímera (el andamiaje no persiste en memoria) y devuelve el
        ``TurnContext`` con las herramientas usadas y las citas recogidas.
        """
        self.history.append(HumanMessage(content=user_message))
        self.last_sources = []
        self.last_images = []
        log_start = len(self.tool_log)

        working = list(self.history)
        final: AIMessage | None = None
        for _ in range(max_tool_rounds):
            ai = self.llm_with_tools.invoke(working)
            if not ai.tool_calls:
                final = ai  # respuesta lista; se reutiliza en ``chat``
                break
            working.append(ai)
            for call in ai.tool_calls:
                working.append(self._execute_tool_call(call))

        return TurnContext(
            working=working,
            tool_calls=self.tool_log[log_start:],
            sources=list(self.last_sources),
            images=list(self.last_images),
            final=final,
        )

    def stream_answer(self, turn: TurnContext) -> Iterator[str]:
        """Emite la respuesta final token a token y la persiste en el historial.

        Usa el LLM SIN herramientas para forzar la generación de texto en lugar
        de nuevas llamadas a tools. Solo el turno conversacional persiste.
        """
        pieces: list[str] = []
        for chunk in self.llm.stream(turn.working):
            text = chunk.content
            if text:
                pieces.append(text)
                yield text

        self.history.append(AIMessage(content="".join(pieces)))
        self._trim_history()

    def stream(self, user_message: str, max_tool_rounds: int = 5) -> Iterator[str]:
        """Resuelve las herramientas y emite la respuesta final token a token."""
        turn = self.prepare(user_message, max_tool_rounds)
        yield from self.stream_answer(turn)

    def chat(self, user_message: str, max_tool_rounds: int = 5) -> dict:
        """Procesa un turno completo y devuelve respuesta, fuentes, fotos y trazas."""
        turn = self.prepare(user_message, max_tool_rounds)

        # Reutiliza la respuesta del bucle si ya se generó (sin tool calls);
        # solo se vuelve a invocar si el bucle terminó pidiendo herramientas.
        final = turn.final if turn.final is not None else self.llm.invoke(turn.working)
        answer = final.content or (
            "Lo siento, no he podido completar la respuesta. Inténtalo de nuevo."
        )

        self.history.append(AIMessage(content=answer))
        self._trim_history()
        return {
            "answer": answer,
            "sources": list(turn.sources),
            "images": list(turn.images),
            "tool_calls": list(turn.tool_calls),
        }

    def _execute_tool_call(self, call: dict) -> ToolMessage:
        """Ejecuta una herramienta, registra la traza y devuelve su ``ToolMessage``.

        Para ``search_tourist_guide`` las citas (fuentes y fotos) llegan en el
        ``artifact`` del ``ToolMessage`` —por llamada— en lugar de leerse de
        estado compartido del RAG (evita mezclar resultados entre sesiones).
        """
        name = call["name"]
        tool = self.tools_by_name[name]
        start = time.perf_counter()
        try:
            message = tool.invoke(call)
            ok = True
        except Exception as exc:  # noqa: BLE001 - resiliencia ante fallos de tool
            message = ToolMessage(
                content=f"Error al ejecutar la herramienta '{name}': {exc}",
                tool_call_id=call["id"],
            )
            ok = False
        elapsed_s = time.perf_counter() - start

        self.tool_log.append(
            ToolCallRecord(
                name=name,
                arguments=dict(call["args"]),
                ok=ok,
                result=str(message.content),
                elapsed_s=elapsed_s,
            )
        )
        logger.info("Herramienta '%s' ejecutada (ok=%s, %.3fs)", name, ok, elapsed_s)

        if name == "search_tourist_guide" and ok:
            artifact = message.artifact or {}
            self.last_sources = list(artifact.get("sources", []))
            self.last_images = list(artifact.get("images", []))

        return message

    def discard_last_user_turn(self) -> None:
        """Descarta el último mensaje de usuario si quedó sin respuesta (p. ej. error).

        Evita dejar un ``HumanMessage`` colgante en el historial cuando un turno
        falla antes de generar la respuesta, lo que rompería turnos posteriores.
        """
        if self.history and isinstance(self.history[-1], HumanMessage):
            self.history.pop()
        self.last_sources = []
        self.last_images = []

    def reset(self) -> None:
        """Reinicia la conversación dejando solo el prompt de sistema."""
        self.history = [SystemMessage(content=build_system_prompt())]
        self.tool_log.clear()
        self.last_sources = []
        self.last_images = []

    def _trim_history(self) -> None:
        """Recorta el historial para mantener la longitud bajo control."""
        self.history = trim_messages(
            self.history,
            token_counter=len,
            max_tokens=self.settings.max_history_messages,
            strategy="last",
            include_system=True,
            start_on="human",
            allow_partial=False,
        )
