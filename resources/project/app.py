"""Aplicación Streamlit del asistente turístico de Tenerife.

Interfaz de chat conversacional que combina RAG sobre la guía oficial
(TENERIFE.pdf), diálogo multiturno y la function call ``get_weather``.
La respuesta se muestra en streaming y se acompaña de las fuentes citadas.
"""

from __future__ import annotations

import streamlit as st

from core.config import load_settings
from core.rag import TouristGuideRAG
from core.assistant import TouristAssistant

# Ejemplos de preguntas que se ofrecen al usuario en la barra lateral.
EXAMPLE_QUESTIONS = [
    "¿Qué playas me recomiendas en el sur de Tenerife?",
    "¿Cómo subir al Teide y qué necesito?",
    "¿Qué tiempo hará el 2026-06-20?",
    "Recomiéndame platos típicos de la gastronomía canaria.",
    "¿Qué puedo visitar en La Laguna?",
]


def get_settings():
    """Carga la configuración del proyecto desde el entorno (.env)."""
    return load_settings()


@st.cache_resource(show_spinner=False)
def get_rag(_settings) -> TouristGuideRAG:
    """Construye (o carga) el índice FAISS una sola vez por sesión de servidor.

    Se usa ``cache_resource`` porque el RAG no tiene estado mutable propio
    de la conversación: solo el índice vectorial, que se reutiliza.
    """
    with st.spinner("Indexando la guía..."):
        rag = TouristGuideRAG(_settings)
        rag.build_index()
    return rag


def render_sources(sources: list[dict]) -> None:
    """Muestra las fuentes citadas dentro de un desplegable."""
    if not sources:
        return
    with st.expander("📚 Fuentes"):
        for i, source in enumerate(sources, start=1):
            page = source.get("page")
            page_label = page + 1 if isinstance(page, int) else "?"
            st.markdown(
                f"**{i}. {source.get('source_name', '?')}** — página {page_label}"
            )
            snippet = source.get("snippet")
            if snippet:
                st.caption(snippet)


def main() -> None:
    """Punto de entrada de la aplicación Streamlit."""
    st.set_page_config(
        page_title="Asistente turístico de Tenerife",
        page_icon="🌴",
    )

    st.title("🌴 Asistente turístico de Tenerife")
    st.caption(
        "Tu guía conversacional para descubrir la isla: playas, rutas, "
        "gastronomía, cultura y el tiempo que hará."
    )

    settings = get_settings()

    # Sin clave de API no podemos hablar con Gemini: avisamos y paramos.
    if not settings.has_api_key:
        st.error(
            "No se ha encontrado la clave de API. Define GOOGLE_API_KEY en "
            "el archivo .env de la raíz del proyecto para usar el asistente."
        )
        st.stop()

    rag = get_rag(settings)

    # El asistente tiene estado mutable (historial), así que vive en la sesión.
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = TouristAssistant(settings, rag)
    assistant: TouristAssistant = st.session_state["assistant"]

    # Historial visible de la conversación.
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # --- Barra lateral: parámetros del modelo, reinicio y ejemplos ---
    with st.sidebar:
        st.header("⚙️ Parámetros del modelo")
        st.write(f"**Modelo:** {settings.generation_model}")
        st.write(f"**Temperature:** {settings.temperature}")
        st.write(f"**Top-p:** {settings.top_p}")
        st.write(f"**Máx. tokens de salida:** {settings.max_output_tokens}")

        if st.button("🔄 Reiniciar conversación"):
            st.session_state["messages"] = []
            assistant.reset()
            st.rerun()

        st.divider()
        st.subheader("💡 Ejemplos de preguntas")
        for question in EXAMPLE_QUESTIONS:
            st.markdown(f"- {question}")

    # --- Render del historial existente ---
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            render_sources(message.get("sources", []))

    # --- Entrada de usuario y respuesta en streaming ---
    prompt = st.chat_input("Escribe tu mensaje...")
    if prompt:
        st.session_state["messages"].append(
            {"role": "user", "content": prompt, "sources": []}
        )
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            answer = st.write_stream(assistant.stream(prompt))
            # Tras la respuesta, el asistente expone las fuentes recogidas.
            sources = list(assistant.last_sources)
            render_sources(sources)

        st.session_state["messages"].append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    main()
