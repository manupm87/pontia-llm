"""Aplicación Streamlit del asistente turístico de Tenerife.

Interfaz de chat conversacional que combina RAG sobre la guía oficial
(TENERIFE.pdf), diálogo multiturno y las function calls de tiempo y estado del
mar. La respuesta se muestra en streaming, con la actividad de las herramientas
en vivo, las fuentes citadas y las fotos de los lugares recuperados de la guía.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import streamlit as st

from core.config import Settings, load_settings
from core.rag import TouristGuideRAG
from core.assistant import TouristAssistant, ToolCallRecord, TurnContext, estimate_cost
from core.guardrails import Guardrails, build_llm_guardrails
from core.photo_match import plan_inline_images
from ui_theme import inject_styles, render_hero

# Ejemplos de preguntas que se ofrecen al usuario (clicables).
EXAMPLE_QUESTIONS = [
    "¿Qué playas me recomiendas en el sur de Tenerife?",
    "¿Cómo subir al Teide y qué necesito?",
    "¿Qué tiempo hará este finde?",
    "¿Está el mar para bañarse mañana?",
    "Recomiéndame platos típicos de la gastronomía canaria.",
    "¿Qué puedo visitar en La Laguna?",
]

# Etiquetas legibles para cada herramienta en el panel de actividad.
_TOOL_LABELS = {
    "search_tourist_guide": "📚 Búsqueda en la guía",
    "get_weather": "🌤️ Previsión del tiempo",
    "get_sea_conditions": "🌊 Estado del mar",
    "resolve_date": "📅 Resolución de fecha",
}


def get_settings() -> Settings:
    """Carga la configuración del proyecto desde el entorno (.env)."""
    return load_settings()


@st.cache_resource(show_spinner=False)
def get_rag(_settings: Settings) -> TouristGuideRAG:
    """Construye (o carga) el índice FAISS una sola vez por sesión de servidor."""
    with st.spinner("Indexando la guía..."):
        rag = TouristGuideRAG(_settings)
        rag.build_index()
    return rag


def get_assistant(
    settings: Settings, rag: TouristGuideRAG, params: tuple[float, float, int, int]
) -> TouristAssistant:
    """Devuelve el asistente, reconstruyéndolo si cambian los parámetros del modelo.

    Reconstruir solo el modelo (no el RAG) permite ajustar la generación en vivo
    conservando el historial de la conversación.
    """
    temperature, top_p, max_tokens, thinking_budget = params
    if (
        "assistant" not in st.session_state
        or st.session_state.get("assistant_params") != params
    ):
        tuned = replace(
            settings,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
            thinking_budget=thinking_budget,
        )
        previous = st.session_state.get("assistant")
        assistant = TouristAssistant(tuned, rag)
        if previous is not None:  # Conserva el historial al reajustar parámetros.
            assistant.history = previous.history
            assistant.tool_log = previous.tool_log
        st.session_state["assistant"] = assistant
        st.session_state["assistant_params"] = params
    return st.session_state["assistant"]


def configure_guardrails(assistant: TouristAssistant, advanced: bool) -> None:
    """Ajusta los guardarraíles del asistente según el modo, sin reconstruir cada rerun.

    Los guardarraíles avanzados envuelven al LLM (instanciar en cada rerun de
    Streamlit sería costoso), así que solo se reconstruyen cuando cambia el modo
    o el asistente. La marca se guarda en la propia instancia del asistente, de
    modo que un asistente nuevo (al reajustar parámetros) recibe unos frescos.
    """
    mode = bool(advanced)
    if getattr(assistant, "_guardrails_mode", None) != mode:
        assistant.guardrails = (
            build_llm_guardrails(assistant.llm) if advanced else Guardrails()
        )
        assistant._guardrails_mode = mode


def render_tool_activity(records: list[ToolCallRecord], container) -> None:
    """Escribe en vivo (dentro del ``st.status``) las herramientas ejecutadas."""
    for record in records:
        label = _TOOL_LABELS.get(record.name, record.name)
        icon = "✅" if record.ok else "⚠️"
        arg = next(iter(record.arguments.values()), "")
        container.write(f"{icon} {label} · `{arg}` · {record.elapsed_s:.2f}s")


def render_tools(tool_calls: list[dict]) -> None:
    """Muestra las herramientas usadas en la respuesta dentro de un desplegable."""
    if not tool_calls:
        return
    with st.expander("🔧 Herramientas usadas"):
        for call in tool_calls:
            label = _TOOL_LABELS.get(call["name"], call["name"])
            icon = "✅" if call["ok"] else "⚠️"
            arg = next(iter(call["arguments"].values()), "")
            st.markdown(f"{icon} **{label}** — `{arg}` ({call['elapsed_s']:.2f}s)")


def render_reasoning(reasoning: str) -> None:
    """Muestra el razonamiento del modelo en un desplegable (colapsado)."""
    if not reasoning:
        return
    with st.expander("🧠 Razonamiento del modelo"):
        st.markdown(reasoning)


def stream_with_reasoning(assistant: TouristAssistant, turn: TurnContext) -> tuple[str, str]:
    """Emite razonamiento y respuesta en vivo; devuelve ``(respuesta, razonamiento)``.

    El razonamiento se va mostrando en directo y, al terminar, se reemplaza por
    un desplegable colapsado encima de la respuesta.
    """
    reasoning_slot = st.empty()
    answer_slot = st.empty()
    thoughts: list[str] = []
    answer: list[str] = []
    for is_thought, text in assistant.stream_reasoning_and_answer(turn):
        if is_thought:
            thoughts.append(text)
            reasoning_slot.markdown("🧠 *Razonando…*\n\n" + "".join(thoughts))
        else:
            answer.append(text)
            answer_slot.markdown("".join(answer))

    reasoning = "".join(thoughts)
    if reasoning:  # Sustituye el razonamiento "en vivo" por un desplegable.
        with reasoning_slot.container():
            render_reasoning(reasoning)

    # Re-renderiza la respuesta intercalando las fotos en sus menciones.
    answer_text = "".join(answer)
    with answer_slot.container():
        render_answer_with_images(answer_text, turn.images)
    return answer_text, reasoning


def render_sources(sources: list[dict]) -> None:
    """Muestra las fuentes citadas dentro de un desplegable."""
    if not sources:
        return
    with st.expander("📚 Fuentes"):
        for i, source in enumerate(sources, start=1):
            page = source.get("page")
            page_label = page + 1 if isinstance(page, int) else "?"
            st.markdown(f"**{i}. {source.get('source_name', '?')}** — página {page_label}")
            text = source.get("snippet")
            if text:
                # Fragmento completo recuperado, como cita en bloque.
                st.markdown("> " + text.replace("\n", "\n> "))


def render_gallery(images: list[dict]) -> None:
    """Renderiza un conjunto de fotos en columnas."""
    if not images:  # ``st.columns(0)`` lanzaría error; nada que mostrar.
        return
    columns = st.columns(min(len(images), 3))
    for index, img in enumerate(images):
        with columns[index % len(columns)]:
            st.image(img["path"], caption=img.get("caption") or "", use_container_width=True)


def render_answer_with_images(text: str, images: list[dict]) -> None:
    """Muestra la respuesta intercalando cada foto donde se menciona su lugar."""
    available = [img for img in images if Path(img.get("path", "")).is_file()]
    for kind, payload in plan_inline_images(text, available):
        if kind == "text":
            if payload.strip():
                st.markdown(payload)
        else:
            render_gallery(payload)


def render_example_buttons(prefix: str, columns: int = 1) -> None:
    """Dibuja los ejemplos como botones que lanzan la pregunta al pulsarlos."""
    cols = st.columns(columns) if columns > 1 else None
    for index, question in enumerate(EXAMPLE_QUESTIONS):
        target = cols[index % columns] if cols else st
        if target.button(question, key=f"{prefix}_{index}"):
            st.session_state["pending_prompt"] = question
            st.rerun()


def records_to_dicts(records: list[ToolCallRecord]) -> list[dict]:
    """Serializa los registros de herramienta para guardarlos en el historial visible."""
    return [
        {
            "name": r.name,
            "arguments": dict(r.arguments),
            "ok": r.ok,
            "elapsed_s": r.elapsed_s,
        }
        for r in records
    ]


def conversation_markdown(messages: list[dict]) -> str:
    """Construye un Markdown descargable con la conversación."""
    lines = ["# Conversación con el asistente turístico de Tenerife", ""]
    for message in messages:
        who = "🧑 Tú" if message["role"] == "user" else "🌴 Asistente"
        lines.append(f"**{who}:** {message['content']}")
        lines.append("")
    return "\n".join(lines)


# Opciones de razonamiento ("thinking") ofrecidas en la barra lateral.
_THINKING_OPTIONS = {
    "Desactivado": 0,
    "Bajo (512)": 512,
    "Medio (1024)": 1024,
    "Alto (4096)": 4096,
    "Dinámico": -1,
}


def render_sidebar(settings: Settings) -> tuple[float, float, int, int, bool, bool]:
    """Dibuja la barra lateral (parámetros, acciones, ejemplos) y devuelve la config.

    Devuelve ``(temperature, top_p, max_tokens, thinking_budget, streaming,
    advanced_guardrails)``.
    """
    with st.sidebar:
        st.header("⚙️ Parámetros del modelo")
        st.caption(f"Modelo: `{settings.generation_model}`")
        temperature = st.slider("Temperature", 0.0, 1.0, settings.temperature, 0.05)
        top_p = st.slider("Top-p", 0.0, 1.0, settings.top_p, 0.05)
        max_tokens = st.slider(
            "Máx. tokens de salida", 256, 4096, settings.max_output_tokens, 128
        )

        st.divider()
        st.subheader("⚡ Respuesta y razonamiento")
        streaming = st.toggle("Respuesta en streaming", value=True)
        labels = list(_THINKING_OPTIONS)
        default_label = next(
            (k for k, v in _THINKING_OPTIONS.items() if v == settings.thinking_budget),
            "Medio (1024)",
        )
        thinking_label = st.selectbox(
            "🧠 Razonamiento (thinking)",
            labels,
            index=labels.index(default_label),
            help="Muestra el razonamiento del modelo en streaming (si lo admite).",
        )
        thinking_budget = _THINKING_OPTIONS[thinking_label]
        advanced_guardrails = st.toggle(
            "🛡️ Guardarraíles avanzados (LLM)",
            value=False,
            help="Bloquea consultas fuera de Tenerife y verifica la fidelidad de la respuesta (consume tokens).",
        )

        # Observabilidad: uso de tokens y coste aproximado acumulados.
        assistant_state = st.session_state.get("assistant")
        if assistant_state is not None:
            usage = assistant_state.total_usage
            cost = estimate_cost(
                usage["input_tokens"], usage["output_tokens"], settings.generation_model
            )
            st.caption(
                f"📊 Tokens: {usage['input_tokens']:,} in · "
                f"{usage['output_tokens']:,} out · ~${cost:.4f}"
            )

        st.divider()
        if st.button("🔄 Reiniciar conversación"):
            st.session_state["messages"] = []
            st.session_state.pop("pending_prompt", None)
            if "assistant" in st.session_state:
                st.session_state["assistant"].reset()
            st.rerun()

        messages = st.session_state.get("messages", [])
        st.download_button(
            "⬇️ Exportar conversación",
            data=conversation_markdown(messages),
            file_name=f"conversacion_tenerife_{datetime.now():%Y%m%d_%H%M}.md",
            mime="text/markdown",
            disabled=not messages,
        )

        st.divider()
        st.subheader("💡 Ejemplos")
        render_example_buttons("ex_side")

    return temperature, top_p, max_tokens, thinking_budget, streaming, advanced_guardrails


def render_history(messages: list[dict]) -> None:
    """Redibuja el historial visible de la conversación."""
    for message in messages:
        with st.chat_message(message["role"]):
            render_reasoning(message.get("reasoning", ""))
            render_answer_with_images(message["content"], message.get("images", []))
            render_sources(message.get("sources", []))
            render_tools(message.get("tool_calls", []))
            render_usage(message.get("usage", {}))


def render_usage(usage: dict) -> None:
    """Muestra el uso de tokens y el coste aproximado de un turno."""
    if not usage:
        return
    st.caption(
        f"🔢 {usage['input_tokens']:,} + {usage['output_tokens']:,} tokens "
        f"· ~${usage['cost']:.4f}"
    )


def handle_turn(
    assistant: TouristAssistant, prompt: str, streaming: bool, advanced: bool
) -> None:
    """Procesa un turno: muestra la actividad de herramientas y la respuesta."""
    st.session_state["messages"].append(
        {"role": "user", "content": prompt, "sources": [], "images": [], "tool_calls": []}
    )
    with st.chat_message("user"):
        st.write(prompt)

    usage_before = dict(assistant.total_usage)
    with st.chat_message("assistant"):
        try:
            # Fase 1: resolución de herramientas, con su actividad en el status.
            with st.status("🧭 Consultando la guía y los datos…", expanded=True) as status:
                turn = assistant.prepare(prompt)
                render_tool_activity(turn.tool_calls, status)
                status.update(label="✅ Datos listos", state="complete", expanded=False)
            # Fase 2: razonamiento + respuesta (en streaming o de una vez).
            if streaming:
                answer, reasoning = stream_with_reasoning(assistant, turn)
            else:
                with st.spinner("✍️ Redactando respuesta…"):
                    answer = assistant.answer(turn)
                reasoning = ""
                render_answer_with_images(answer, turn.images)
        except Exception as exc:  # noqa: BLE001 - mostrar el fallo sin romper la app
            st.error(f"No he podido completar la respuesta: {exc}")
            # Deshace el turno a medias para no corromper los siguientes.
            assistant.discard_last_user_turn()
            st.session_state["messages"].pop()
            return

        # Guardarraíl de salida (opcional): avisa si la respuesta no se apoya en la guía.
        if advanced and not turn.blocked and turn.sources:
            context = "\n\n".join(s.get("snippet", "") for s in turn.sources)
            if not assistant.guardrails.check_output(answer, context).allowed:
                st.warning("⚠️ No he podido verificar del todo esta información en la guía oficial.")

        tool_calls = records_to_dicts(turn.tool_calls)
        render_sources(turn.sources)
        render_tools(tool_calls)

        usage = {
            "input_tokens": assistant.total_usage["input_tokens"] - usage_before["input_tokens"],
            "output_tokens": assistant.total_usage["output_tokens"] - usage_before["output_tokens"],
        }
        usage["cost"] = estimate_cost(
            usage["input_tokens"], usage["output_tokens"], assistant.settings.generation_model
        )
        render_usage(usage)

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "reasoning": reasoning,
            "sources": turn.sources,
            "images": turn.images,
            "tool_calls": tool_calls,
            "usage": usage,
        }
    )


def main() -> None:
    """Punto de entrada de la aplicación Streamlit."""
    st.set_page_config(page_title="Asistente turístico de Tenerife", page_icon="🌴")
    inject_styles()
    render_hero()

    settings = get_settings()

    # Sin clave de API no podemos hablar con Gemini: avisamos y paramos.
    if not settings.has_api_key:
        st.error(
            "No se ha encontrado la clave de API. Define GOOGLE_API_KEY en "
            "el archivo .env de la raíz del proyecto para usar el asistente."
        )
        st.stop()

    rag = get_rag(settings)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    temperature, top_p, max_tokens, thinking_budget, streaming, advanced = render_sidebar(settings)
    assistant = get_assistant(settings, rag, (temperature, top_p, max_tokens, thinking_budget))
    configure_guardrails(assistant, advanced)

    render_history(st.session_state["messages"])

    # Entrada del usuario: del chat o de un ejemplo pulsado. El ejemplo pendiente
    # se extrae siempre (no debe quedar "atascado" para reaparecer más tarde).
    pending = st.session_state.pop("pending_prompt", None)
    prompt = st.chat_input("Escribe tu mensaje...") or pending

    # Estado de bienvenida: invita a probar con ejemplos clicables.
    if not st.session_state["messages"] and not prompt:
        st.markdown("#### ✨ Empieza por aquí")
        render_example_buttons("ex_home", columns=2)

    if prompt:
        handle_turn(assistant, prompt, streaming, advanced)


if __name__ == "__main__":
    main()
