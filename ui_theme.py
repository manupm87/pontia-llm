"""Tema visual de la aplicación Streamlit (CSS dinámico y cabecera animada).

Reúne el estilo "premium" del asistente: una paleta inspirada en Tenerife
(océano atlántico + volcán del Teide + arena), tipografías de Google Fonts,
fondo con auroras animadas, tarjetas tipo *glassmorphism* y micro-animaciones.
Se mantiene aparte de ``app.py`` para no mezclar presentación con lógica.

Streamlit no expone clases estables, así que el CSS se apoya en los
``data-testid`` documentados (contenedor, barra lateral, chat, expander, etc.).
"""

from __future__ import annotations

import streamlit as st

# Paleta de la marca (océano, volcán y arena de Tenerife).
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Poppins:wght@600;700;800&display=swap');

:root {
    --ocean-1: #0ea5b7;   /* turquesa atlántico */
    --ocean-2: #1e6fd9;   /* azul profundo */
    --volcano-1: #ff7a45; /* lava cálida */
    --volcano-2: #ff4d6d; /* magma */
    --sand: #fff7ed;      /* arena clara */
    --ink: #0f2233;       /* texto */
    --glass: rgba(255, 255, 255, 0.62);
    --glass-brd: rgba(255, 255, 255, 0.55);
    --shadow: 0 10px 30px -12px rgba(15, 34, 51, 0.35);
}

/* ---- Lienzo general con auroras animadas ---- */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(1100px 600px at 8% -8%, rgba(14,165,183,0.18), transparent 60%),
        radial-gradient(900px 520px at 105% 6%, rgba(255,122,69,0.16), transparent 60%),
        radial-gradient(800px 600px at 50% 115%, rgba(30,111,217,0.14), transparent 60%),
        linear-gradient(180deg, #f7fbff 0%, #fef6f0 100%);
    background-attachment: fixed;
}
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: -20% -10% auto -10%;
    height: 60vh;
    background:
        radial-gradient(40% 40% at 20% 30%, rgba(14,165,183,0.30), transparent 70%),
        radial-gradient(38% 38% at 80% 20%, rgba(255,77,109,0.22), transparent 70%);
    filter: blur(40px);
    z-index: 0;
    animation: aurora 18s ease-in-out infinite alternate;
    pointer-events: none;
}
@keyframes aurora {
    0%   { transform: translate3d(0,0,0) scale(1);   opacity: 0.85; }
    50%  { transform: translate3d(3%, 2%, 0) scale(1.08); opacity: 1; }
    100% { transform: translate3d(-3%, -1%, 0) scale(1.04); opacity: 0.9; }
}

/* ---- Cabecera "hero" con degradado animado ---- */
.hero {
    position: relative;
    z-index: 1;
    margin: 0.2rem 0 1.4rem 0;
    padding: 1.6rem 1.8rem;
    border-radius: 22px;
    background: var(--glass);
    border: 1px solid var(--glass-brd);
    box-shadow: var(--shadow);
    backdrop-filter: blur(14px);
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--ocean-1), var(--ocean-2), var(--volcano-1), var(--volcano-2), var(--ocean-1));
    background-size: 300% 100%;
    animation: slide 6s linear infinite;
}
@keyframes slide { to { background-position: 300% 0; } }
.hero h1 {
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 2.1rem;
    line-height: 1.1;
    margin: 0;
}
/* Solo el texto lleva el degradado recortado; el emoji conserva su color. */
.hero h1 .grad {
    background: linear-gradient(100deg, var(--ocean-2), var(--ocean-1) 40%, var(--volcano-1) 80%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    font-family: 'Inter', sans-serif;
    color: #335; margin: 0.45rem 0 0 0; font-size: 1rem;
}
.hero .badges { margin-top: 0.9rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
.hero .badge {
    font-family: 'Inter', sans-serif; font-size: 0.78rem; font-weight: 600;
    color: var(--ink); padding: 0.32rem 0.7rem; border-radius: 999px;
    background: rgba(255,255,255,0.7); border: 1px solid var(--glass-brd);
}

/* ---- Tipografía base (por herencia; no se usa ``*`` para no romper los
   glifos de los iconos Material de Streamlit, que llevan su propia fuente). ---- */
[data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
    font-family: 'Inter', sans-serif;
}

/* ---- Burbujas de chat tipo glass con animación de entrada ---- */
[data-testid="stChatMessage"] {
    position: relative; z-index: 1;
    background: var(--glass);
    border: 1px solid var(--glass-brd);
    border-radius: 18px;
    padding: 0.5rem 0.9rem;
    box-shadow: var(--shadow);
    backdrop-filter: blur(10px);
    animation: rise 0.4s cubic-bezier(.2,.8,.2,1) both;
}
[data-testid="stChatMessage"]::before {
    content: ""; position: absolute; left: 0; top: 14px; bottom: 14px; width: 4px;
    border-radius: 4px;
    background: linear-gradient(180deg, var(--ocean-1), var(--ocean-2));
}
@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }

/* ---- Botones (ejemplos y acciones) ---- */
.stButton > button {
    width: 100%;
    border: 1px solid var(--glass-brd);
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(14,165,183,0.12), rgba(255,122,69,0.12));
    color: var(--ink);
    font-weight: 600;
    text-align: left;
    padding: 0.6rem 0.85rem;
    transition: transform .15s ease, box-shadow .2s ease, background .2s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 24px -10px rgba(14,165,183,0.55);
    background: linear-gradient(135deg, rgba(14,165,183,0.22), rgba(255,122,69,0.22));
    border-color: var(--ocean-1);
}
.stButton > button:active { transform: translateY(0); }

/* ---- Barra lateral ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.85), rgba(247,251,255,0.85));
    border-right: 1px solid var(--glass-brd);
    backdrop-filter: blur(12px);
}

/* ---- Expanders (fuentes, fotos, herramientas) ---- */
[data-testid="stExpander"] {
    border: 1px solid var(--glass-brd);
    border-radius: 16px;
    background: var(--glass);
    box-shadow: var(--shadow);
    overflow: hidden;
}

/* ---- Imágenes de la guía con esquinas suaves ---- */
[data-testid="stImage"] img {
    border-radius: 14px;
    box-shadow: var(--shadow);
}

/* ---- Caja de entrada del chat tipo "pill" con glow al enfocar ---- */
[data-testid="stChatInput"] {
    border-radius: 16px;
    border: 1px solid var(--glass-brd);
    background: var(--glass);
    box-shadow: var(--shadow);
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--ocean-1);
    box-shadow: 0 0 0 3px rgba(14,165,183,0.25);
}

/* Oculta el encabezado por defecto de Streamlit para lucir el hero. */
[data-testid="stHeader"] { background: transparent; }
</style>
"""

# Cabecera principal con título degradado y "chips" de capacidades.
_HERO_HTML = """
<div class="hero">
  <h1>🌴 <span class="grad">Asistente turístico de Tenerife</span></h1>
  <p>Tu guía conversacional para descubrir la isla: playas, rutas, gastronomía,
  cultura, el tiempo y el estado del mar.</p>
  <div class="badges">
    <span class="badge">📚 Guía oficial (RAG)</span>
    <span class="badge">🌤️ Tiempo</span>
    <span class="badge">🌊 Estado del mar</span>
    <span class="badge">📅 Fechas inteligentes</span>
    <span class="badge">🖼️ Fotos de lugares</span>
  </div>
</div>
"""


def inject_styles() -> None:
    """Inyecta el CSS del tema en la página."""
    st.markdown(_CSS, unsafe_allow_html=True)


def render_hero() -> None:
    """Dibuja la cabecera animada del asistente."""
    st.markdown(_HERO_HTML, unsafe_allow_html=True)
