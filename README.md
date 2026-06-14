# Asistente turístico de Tenerife

Asistente conversacional que ayuda a los turistas a planificar su viaje a
Tenerife. Responde en español, de forma cercana y práctica, combinando estas
capacidades:

- **RAG sobre la guía oficial** (`data/TENERIFE.pdf`): recupera información sobre
  playas, rutas, gastronomía, cultura y lugares de interés, citando siempre la
  página de origen y **basándose solo en los fragmentos recuperados** (no en el
  conocimiento previo del modelo). La guía se consulta **en cada turno**, así que
  las fuentes se muestran siempre, también en preguntas de seguimiento.
- **Fotos de los lugares**: extrae las fotografías embebidas en el PDF y muestra
  las del lugar recuperado junto a la respuesta.
- **Consulta meteorológica** (`get_weather`): obtiene la previsión del tiempo en
  Tenerife para una fecha concreta a través de la API pública de Open-Meteo.
- **Estado del mar** (`get_sea_conditions`): consulta el oleaje (altura y periodo)
  en la API marina de Open-Meteo para saber si el mar está apto para el baño o el
  surf, con respaldo simulado ante fallos de red.
- **Fechas inteligentes** (`resolve_date`): traduce expresiones relativas ("hoy",
  "mañana", "este finde", "el miércoles") a una fecha exacta `YYYY-MM-DD` para las
  herramientas de tiempo y mar. El prompt de sistema se ancla a la fecha actual.
- **Diálogo multiturno**: mantiene el contexto de la conversación para resolver
  preguntas encadenadas.
- **Razonamiento en streaming**: si el modelo lo admite, muestra el
  razonamiento ("thinking" de Gemini) y la respuesta en *streaming* en vivo.
- **Interfaz cuidada**: tema visual propio (CSS dinámico), ejemplos clicables,
  actividad de las herramientas en vivo, fuentes con el fragmento completo
  citado, galería de fotos y exportación del chat.

El asistente está construido sobre **Google Gemini** orquestado con
**LangChain**, usa un **vector store FAISS** para la búsqueda semántica y
**PyMuPDF** para extraer las imágenes de la guía.

## Requisitos del enunciado que cubre

- **RAG expuesto como herramienta**: la búsqueda en la guía no está cableada de
  forma rígida en el flujo, sino que se ofrece al modelo como una *tool*
  (`search_tourist_guide`) que decide cuándo invocar.
- **Function calling**: funciones externas (`get_weather`, `get_sea_conditions`)
  integradas como herramientas que el LLM puede llamar, además de `resolve_date`
  para normalizar fechas relativas.
- **Conversación multiturno**: gestión de historial con recorte automático para
  mantener el contexto bajo control.
- **Stack Gemini + LangChain** con **FAISS** como vector store.
- **Citas de fuentes**: cada respuesta basada en el documento incluye nombre de
  archivo, página y fragmento.
- **Respuestas ancladas al documento (*grounding*)**: el prompt obliga a consultar
  la guía en cada turno y a responder solo con lo recuperado, evitando que el
  modelo conteste de memoria.
- **Multimodalidad de la guía**: las fotos del PDF se muestran junto a la respuesta.
- **Despliegue**: interfaz web con Streamlit y respuesta en *streaming* real de
  tokens.

## Arquitectura

```
Usuario ──▶ TouristAssistant (LangChain + Gemini)
                │
                │  bind_tools + bucle de tool calling
                ▼
        ┌───────────────┬────────────────┐
        ▼               ▼                 │
 search_tourist_guide  get_weather        │
        │               │                 │
        ▼               ▼                 ▼
   TouristGuideRAG   Open-Meteo      Historial multiturno
   (FAISS + Gemini    (requests)     (trim_messages)
    embeddings)
```

- **RAG como herramienta**: `TouristGuideRAG` (en `core/rag.py`) carga el PDF, lo
  trocea con solapamiento, genera embeddings con Gemini y construye el índice
  FAISS. La recuperación devuelve el contexto formateado, las fuentes citadas y las
  fotos de las páginas recuperadas. Se expone al modelo como la herramienta
  `search_tourist_guide`.
- **Fotos de la guía**: `GuideImageStore` (en `core/images.py`) extrae con
  **PyMuPDF** las imágenes embebidas en el PDF, las guarda en `storage/images/` y
  las indexa por página, de modo que el asistente puede mostrar la foto del lugar
  recuperado en cada respuesta.
- **Anclaje al documento y citas en cada turno**: el andamiaje de *tool calling*
  (peticiones y resultados de herramientas) no se guarda en la memoria; solo
  persisten los turnos de usuario y asistente. Así el modelo vuelve a consultar la
  guía en cada pregunta, refresca las fuentes y las fotos, y no responde de memoria.
- **get_weather**: `core/weather.py` consulta Open-Meteo para las coordenadas de
  Tenerife y devuelve temperaturas máxima/mínima y precipitación. Se expone como
  la herramienta `get_weather`, que recibe la fecha en formato `YYYY-MM-DD`.
- **Multiturno**: `TouristAssistant` (en `core/assistant.py`) mantiene el
  historial de mensajes, ejecuta el bucle de *tool calling* (varias rondas hasta
  obtener la respuesta final) y recorta el historial con `trim_messages`.
- **Gemini + LangChain**: el modelo se inicializa con `init_chat_model` y se le
  asocian las herramientas con `bind_tools`. El prompt de sistema fija el rol del
  asistente y las reglas de uso de cada herramienta.
- **FAISS**: índice vectorial persistente en disco; se carga si ya existe y se
  reconstruye solo cuando es necesario.

## Estructura del proyecto

```
.
├── app.py                     # Interfaz web con Streamlit
├── ui_theme.py                # Tema visual (CSS dinámico + cabecera animada)
├── core/                      # Paquete reutilizable del asistente
│   ├── __init__.py
│   ├── config.py              # Settings inmutable y carga desde el entorno
│   ├── weather.py             # Función externa get_weather (Open-Meteo)
│   ├── sea.py                 # Función externa get_sea_conditions (Open-Meteo marino)
│   ├── dates.py               # resolve_date: fechas relativas -> YYYY-MM-DD
│   ├── rag.py                 # Indexado del PDF y recuperación con citas
│   ├── images.py              # Extracción de las fotos del PDF (por página)
│   ├── tools.py               # Herramientas LangChain (@tool) para el LLM
│   └── assistant.py           # Asistente conversacional (chat / stream)
├── tests/                     # Suite de tests (pytest)
├── data/
│   └── TENERIFE.pdf           # Guía oficial usada como fuente del RAG
├── storage/
│   ├── faiss_index/           # Índice FAISS (se genera en la primera ejecución)
│   └── images/                # Fotos extraídas del PDF + manifest.json
├── notebook_asistente_tenerife.ipynb   # Notebook de demostración
├── requirements.txt
├── .env.template              # Plantilla de variables de entorno
└── README.md
```

## Instalación

Crea un entorno virtual e instala las dependencias:

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuración de la API key

Copia la plantilla de variables de entorno y rellena tu clave de Google:

```bash
cp .env.template .env
```

Edita el archivo `.env` y asigna tu clave de la API de Google (Gemini):

```
GOOGLE_API_KEY="tu_clave_aqui"
```

La clave se carga automáticamente con `python-dotenv` y la detecta
`langchain-google-genai`.

## Cómo ejecutarlo

### Notebook de demostración

```bash
jupyter lab notebook_asistente_tenerife.ipynb
```

El notebook muestra el flujo completo: construcción del índice, recuperación con
citas, llamadas a herramientas y conversación multiturno.

### Aplicación web

```bash
streamlit run app.py
```

La app ofrece el chat con el asistente, preguntas de ejemplo clicables, respuesta
en *streaming*, la actividad de las herramientas en vivo, la galería de fotos de
la guía, el panel de fuentes citadas y la exportación de la conversación. Desde
la barra lateral se ajustan **en vivo** los parámetros del modelo (temperatura,
top-p, tokens), el **streaming** y el **razonamiento (thinking)**.

## Tests

La lógica reutilizable (fechas, estado del mar, herramientas y orquestación del
asistente) está cubierta por una suite de `pytest` que no requiere red ni las
dependencias pesadas (usa dobles de prueba para el LLM y el RAG):

```bash
python -m pytest
```

## Parámetros del modelo configurables por entorno

Estos parámetros pueden ajustarse mediante variables de entorno (por ejemplo en
`.env`); si no se definen, se usan los valores por defecto:

| Variable             | Valor por defecto              | Descripción                          |
|----------------------|--------------------------------|--------------------------------------|
| `GOOGLE_API_KEY`     | —                              | Clave de la API de Google (Gemini)   |
| `GENERATION_MODEL`   | `gemini-2.5-flash-lite`        | Modelo de generación                 |
| `EMBEDDING_MODEL`    | `models/gemini-embedding-001`  | Modelo de embeddings                 |
| `TEMPERATURE`        | `0.2`                          | Temperatura de muestreo              |
| `TOP_P`              | `0.95`                         | Núcleo de probabilidad (top-p)       |
| `MAX_OUTPUT_TOKENS`  | `1024`                         | Límite de tokens de la respuesta     |
| `THINKING_BUDGET`    | `1024`                         | Presupuesto de razonamiento de Gemini (0 = off, -1 = dinámico) |

El resto de parámetros (tamaño de troceado `chunk_size=500`, solapamiento
`chunk_overlap=100`, `top_k=5` de recuperación, longitud máxima del historial,
rutas del PDF y del índice, tamaño mínimo de imagen y número máximo de fotos por
respuesta) se definen en `core/config.py` mediante la clase `Settings`.

## Índice FAISS e imágenes

El índice vectorial y las **fotos de la guía** se **construyen en la primera
ejecución** a partir de `data/TENERIFE.pdf` y se persisten en
`storage/faiss_index/` y `storage/images/` (con un `manifest.json` que las mapea
por página). En las ejecuciones siguientes se cargan directamente desde disco,
por lo que el arranque es mucho más rápido. Para forzar su reconstrucción (por
ejemplo tras actualizar el PDF o cambiar el modelo de embeddings), elimina el
directorio `storage/` o invoca `build_index(force=True)`.

## Limitaciones y mejoras futuras

- **Cobertura limitada al PDF**: el asistente solo conoce lo que contiene la guía
  oficial; si la información no está en el documento, lo indica en lugar de
  inventarla.
- **Previsión meteorológica acotada**: Open-Meteo ofrece pronóstico para un rango
  de días limitado; para fechas muy lejanas la respuesta puede ser aproximada o
  no estar disponible.
- **Memoria por recorte**: el historial se trunca para controlar el coste, de
  modo que las conversaciones muy largas pueden perder contexto antiguo.

Posibles mejoras:

- Añadir más documentos y fuentes (eventos, transporte, horarios actualizados).
- Incorporar *reranking* y evaluación automática de la calidad de la
  recuperación.
- Asociar las fotos a los lugares por análisis del *layout* (no solo por página) o
  con pies de foto generados por un modelo multimodal.
- Resumir el historial antiguo en lugar de recortarlo para conservar contexto.
- Soporte multilingüe para turistas que no hablen español.
- Persistencia de conversaciones y caché de respuestas frecuentes.
```
