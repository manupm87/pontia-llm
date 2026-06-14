"""Asistente turístico conversacional de Tenerife.

Orquesta el diálogo multiturno con Gemini: gestiona el historial, ejecuta el
bucle de *tool calling* (RAG sobre la guía y consulta meteorológica), expone
una respuesta completa con ``chat`` y una respuesta en *streaming* real de
tokens con ``stream``. Además mantiene un registro de las herramientas usadas
para observabilidad.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterator

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)

from .config import Settings
from .rag import TouristGuideRAG
from . import tools as tools_module

logger = logging.getLogger("asistente_tenerife.assistant")

# Prompt de sistema en español: define el rol y las reglas del asistente.
SYSTEM_PROMPT = (
    "Eres un asistente turístico experto en la isla de Tenerife. Ayudas a los "
    "turistas a planificar su viaje de forma cercana y útil.\n\n"
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
    "- Usa la herramienta 'get_weather' para informar del tiempo en una fecha "
    "concreta (formato YYYY-MM-DD).\n"
    "- Responde siempre en español, con un tono cercano y práctico."
)


@dataclass
class ToolCallRecord:
    """Registro de una llamada a herramienta para observabilidad y trazas."""

    name: str
    arguments: dict
    ok: bool
    result: str
    elapsed_s: float


class TouristAssistant:
    """Asistente conversacional con memoria, RAG y *function calling*."""

    def __init__(self, settings: Settings, rag: TouristGuideRAG) -> None:
        """Inicializa el LLM con herramientas y arranca el historial."""
        # La herramienta de RAG necesita la instancia compartida del índice.
        tools_module.set_rag_instance(rag)

        self.settings = settings
        self.rag = rag
        self.llm = init_chat_model(
            settings.generation_model,
            model_provider="google_genai",
            temperature=settings.temperature,
            top_p=settings.top_p,
            max_tokens=settings.max_output_tokens,
        )
        self.tools = tools_module.get_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tools_by_name = {t.name: t for t in self.tools}
        self.history: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
        self.tool_log: list[ToolCallRecord] = []
        # Fuentes y fotos citadas en la última respuesta (para Streamlit).
        self.last_sources: list[dict] = []
        self.last_images: list[dict] = []

    def _run_tool_rounds(
        self, working: list[BaseMessage], max_tool_rounds: int
    ) -> AIMessage | None:
        """Ejecuta rondas de *tool calling* hasta obtener una respuesta final.

        Invoca el modelo con herramientas sobre ``working`` (una copia efímera
        del historial); si no pide ninguna, devuelve ese mensaje como respuesta
        final. Si pide herramientas, las ejecuta, añade sus resultados a
        ``working`` y repite. Devuelve ``None`` si se agotan las rondas sin
        respuesta final.
        """
        for _ in range(max_tool_rounds):
            ai = self.llm_with_tools.invoke(working)

            # Sin llamadas a herramientas: el modelo ya tiene la respuesta.
            if not ai.tool_calls:
                return ai

            working.append(ai)
            for call in ai.tool_calls:
                working.append(self._execute_tool_call(call))

        return None

    def _execute_tool_call(self, call: dict) -> ToolMessage:
        """Ejecuta una herramienta, registra la traza y devuelve su ``ToolMessage``."""
        name = call["name"]
        tool = self.tools_by_name[name]
        start = time.perf_counter()
        try:
            result = tool.invoke(call["args"])
            ok = True
        except Exception as exc:  # noqa: BLE001 - resiliencia ante fallos de tool
            result = f"Error al ejecutar la herramienta '{name}': {exc}"
            ok = False
        elapsed_s = time.perf_counter() - start

        record = ToolCallRecord(
            name=name,
            arguments=dict(call["args"]),
            ok=ok,
            result=str(result),
            elapsed_s=elapsed_s,
        )
        self.tool_log.append(record)
        logger.info(
            "Herramienta '%s' ejecutada (ok=%s, %.3fs)", name, ok, elapsed_s
        )

        # La búsqueda en la guía deja sus fuentes y fotos citadas en el RAG.
        if name == "search_tourist_guide":
            self.last_sources = list(self.rag.last_sources)
            self.last_images = list(self.rag.last_images)

        return ToolMessage(content=str(result), tool_call_id=call["id"])

    def chat(self, user_message: str, max_tool_rounds: int = 5) -> dict:
        """Procesa un turno completo y devuelve respuesta, fuentes, fotos y trazas."""
        self.history.append(HumanMessage(content=user_message))
        # Reinicia los acumuladores de fuentes/fotos y marca el inicio de trazas.
        self.last_sources = []
        self.last_images = []
        tool_log_start = len(self.tool_log)

        # El andamiaje de *tool calling* (peticiones y resultados de herramientas)
        # es efímero: se trabaja sobre una copia para no dejarlo en la memoria.
        working = list(self.history)
        final = self._run_tool_rounds(working, max_tool_rounds)
        answer = final.content if final else (
            "Lo siento, no he podido completar la respuesta. Inténtalo de nuevo."
        )

        # En la memoria solo persiste el turno conversacional (sin herramientas),
        # de modo que cada nueva pregunta vuelve a consultar la guía y refresca
        # sus fuentes e imágenes.
        self.history.append(AIMessage(content=answer))
        self._trim_history()
        return {
            "answer": answer,
            "sources": list(self.last_sources),
            "images": list(self.last_images),
            "tool_calls": self.tool_log[tool_log_start:],
        }

    def stream(self, user_message: str, max_tool_rounds: int = 5) -> Iterator[str]:
        """Resuelve las herramientas y emite la respuesta final token a token."""
        self.history.append(HumanMessage(content=user_message))
        self.last_sources = []
        self.last_images = []

        # Andamiaje de herramientas sobre una copia efímera del historial (no se
        # puede stremear de forma fiable mientras se decide llamar a una tool).
        working = list(self.history)
        for _ in range(max_tool_rounds):
            ai = self.llm_with_tools.invoke(working)
            if not ai.tool_calls:
                # Sin herramientas: rompemos y generamos la respuesta final.
                break
            working.append(ai)
            for call in ai.tool_calls:
                working.append(self._execute_tool_call(call))

        # Respuesta final en streaming real con el LLM SIN herramientas, para
        # forzar la generación de texto en lugar de nuevas llamadas a tools.
        pieces: list[str] = []
        for chunk in self.llm.stream(working):
            text = chunk.content
            if text:
                pieces.append(text)
                yield text

        # Solo el turno conversacional persiste en la memoria (ver ``chat``).
        full_text = "".join(pieces)
        self.history.append(AIMessage(content=full_text))
        self._trim_history()

    def reset(self) -> None:
        """Reinicia la conversación dejando solo el prompt de sistema."""
        self.history = [SystemMessage(content=SYSTEM_PROMPT)]
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
