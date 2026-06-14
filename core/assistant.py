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
        "también en las preguntas de seguimiento.\n"
        "- Usa ÚNICAMENTE la información que devuelvan las herramientas; no añadas "
        "lugares ni datos que no aparezcan en ella. Si no tienes la información, "
        "dilo con naturalidad y no inventes.\n"
        "- Habla en primera persona, como un guía local experto que conoce la isla "
        "de memoria. NO menciones 'la guía', 'el documento', 'los fragmentos' ni "
        "cites números de página: comparte el conocimiento como propio, de forma "
        "natural. Las fuentes se muestran aparte, no hace falta que las nombres.\n"
        "- Cuando el usuario mencione una fecha relativa (hoy, mañana, este finde, "
        "el miércoles...), usa primero 'resolve_date' para obtener la fecha exacta "
        "(YYYY-MM-DD) y pásala a 'get_weather' o 'get_sea_conditions'.\n"
        "- Usa 'get_weather' para la previsión meteorológica y 'get_sea_conditions' "
        "para el estado del mar (oleaje, baño, surf) en una fecha concreta.\n"
        "- Razona y piensa SIEMPRE en español: tanto tu razonamiento interno "
        "como la respuesta final deben estar en español.\n"
        "- Responde siempre en español, con un tono cercano y práctico."
    )


# Prompt por defecto (anclado a la fecha de import); se mantiene por
# compatibilidad con el notebook. El asistente lo reconstruye en cada sesión.
SYSTEM_PROMPT = build_system_prompt()

# Claves que, en los bloques de contenido de Gemini, marcan un fragmento de
# razonamiento ("thinking") en lugar de respuesta.
_THOUGHT_TYPES = {"thinking", "reasoning", "thought"}


def _split_content(chunk) -> list[tuple[bool, str]]:
    """Separa un *chunk* de streaming en fragmentos ``(is_thought, text)``.

    Tolera las distintas formas en que el SDK puede entregar el contenido
    (cadena simple, lista de bloques con tipo, o razonamiento en
    ``additional_kwargs``); si no detecta razonamiento, todo es respuesta.
    """
    out: list[tuple[bool, str]] = []

    # Algunos proveedores entregan el razonamiento aparte del contenido.
    reasoning = getattr(chunk, "additional_kwargs", {}).get("reasoning_content")
    if reasoning:
        out.append((True, str(reasoning)))

    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        if content:
            out.append((False, content))
        return out

    for block in content or []:
        if isinstance(block, str):
            if block:
                out.append((False, block))
        elif isinstance(block, dict):
            is_thought = block.get("type") in _THOUGHT_TYPES or bool(block.get("thought"))
            text = (
                block.get("thinking")
                or block.get("reasoning")
                or block.get("text")
                or ""
            )
            if text:
                out.append((is_thought, text))
    return out


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
        """Crea el modelo de Gemini (import perezoso para no exigir ``langchain``).

        Si ``thinking_budget`` lo permite, activa el "thinking" de Gemini con
        resúmenes de razonamiento. Si la versión del SDK no admite esos
        parámetros, se reintenta sin ellos para no romper el arranque.
        """
        from langchain.chat_models import init_chat_model

        common = dict(
            model_provider="google_genai",
            temperature=settings.temperature,
            top_p=settings.top_p,
            max_tokens=settings.max_output_tokens,
        )
        # Intenta activar el "thinking" con resúmenes; los nombres de parámetros
        # varían entre versiones del SDK, así que se prueban en cascada y, como
        # último recurso, se arranca sin razonamiento.
        attempts = []
        if settings.thinking_budget != 0:
            attempts.append(
                dict(thinking_budget=settings.thinking_budget, include_thoughts=True)
            )
            attempts.append(dict(thinking_budget=settings.thinking_budget))
        attempts.append({})
        for extra in attempts:
            try:
                return init_chat_model(settings.generation_model, **common, **extra)
            except Exception as exc:  # noqa: BLE001 - prueba la siguiente combinación
                logger.warning("init_chat_model falló con %s (%s); reintentando.", extra, exc)
        # Si todo falla, propaga el último error construyendo sin extras.
        return init_chat_model(settings.generation_model, **common)

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

    def stream_reasoning_and_answer(
        self, turn: TurnContext
    ) -> Iterator[tuple[bool, str]]:
        """Emite el razonamiento y la respuesta en *streaming*.

        Genera tuplas ``(is_thought, text)``: ``True`` para los fragmentos de
        razonamiento ("thinking" de Gemini) y ``False`` para la respuesta. Solo
        la respuesta (sin el razonamiento) se persiste en el historial.
        """
        answer: list[str] = []
        for chunk in self.llm.stream(turn.working):
            for is_thought, text in _split_content(chunk):
                if not is_thought:
                    answer.append(text)
                yield is_thought, text

        self.history.append(AIMessage(content="".join(answer)))
        self._trim_history()

    def stream_answer(self, turn: TurnContext) -> Iterator[str]:
        """Emite solo la respuesta final token a token (sin el razonamiento).

        Se conserva por compatibilidad (notebook). Usa el LLM SIN herramientas
        para forzar la generación de texto en lugar de nuevas llamadas a tools.
        """
        for is_thought, text in self.stream_reasoning_and_answer(turn):
            if not is_thought:
                yield text

    def stream(self, user_message: str, max_tool_rounds: int = 5) -> Iterator[str]:
        """Resuelve las herramientas y emite la respuesta final token a token."""
        turn = self.prepare(user_message, max_tool_rounds)
        yield from self.stream_answer(turn)

    def answer(self, turn: TurnContext) -> str:
        """Devuelve la respuesta final (sin streaming) de un turno ya preparado.

        Reutiliza la respuesta del bucle si ya se generó (sin tool calls); solo
        vuelve a invocar si el bucle terminó pidiendo herramientas. Descarta el
        razonamiento y persiste solo la respuesta en el historial.
        """
        final = turn.final if turn.final is not None else self.llm.invoke(turn.working)
        text = "".join(t for is_thought, t in _split_content(final) if not is_thought) or (
            "Lo siento, no he podido completar la respuesta. Inténtalo de nuevo."
        )
        self.history.append(AIMessage(content=text))
        self._trim_history()
        return text

    def chat(self, user_message: str, max_tool_rounds: int = 5) -> dict:
        """Procesa un turno completo y devuelve respuesta, fuentes, fotos y trazas."""
        turn = self.prepare(user_message, max_tool_rounds)
        return {
            "answer": self.answer(turn),
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
