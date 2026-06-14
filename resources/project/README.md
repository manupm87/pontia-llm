# Asistente turístico de Tenerife

Asistente conversacional que ayuda a los turistas a planificar su viaje a
Tenerife. Responde en español, de forma cercana y práctica, combinando tres
capacidades:

- **RAG sobre la guía oficial** (`data/TENERIFE.pdf`): recupera información sobre
  playas, rutas, gastronomía, cultura y lugares de interés, citando siempre la
  página de origen.
- **Consulta meteorológica** (`get_weather`): obtiene la previsión del tiempo en
  Tenerife para una fecha concreta a través de la API pública de Open-Meteo.
- **Diálogo multiturno**: mantiene el contexto de la conversación para resolver
  preguntas encadenadas.

El asistente está construido sobre **Google Gemini** orquestado con
**LangChain**, y usa un **vector store FAISS** para la búsqueda semántica.

## Requisitos del enunciado que cubre

- **RAG expuesto como herramienta**: la búsqueda en la guía no está cableada de
  forma rígida en el flujo, sino que se ofrece al modelo como una *tool*
  (`search_tourist_guide`) que decide cuándo invocar.
- **Function calling**: una función externa `get_weather` integrada también como
  herramienta que el LLM puede llamar.
- **Conversación multiturno**: gestión de historial con recorte automático para
  mantener el contexto bajo control.
- **Stack Gemini + LangChain** con **FAISS** como vector store.
- **Citas de fuentes**: cada respuesta basada en el documento incluye nombre de
  archivo, página y fragmento.
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
  FAISS. La recuperación devuelve el contexto formateado y las fuentes citadas.
  Se expone al modelo como la herramienta `search_tourist_guide`.
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
project/
├── app.py                     # Interfaz web con Streamlit
├── core/                      # Paquete reutilizable del asistente
│   ├── __init__.py
│   ├── config.py              # Settings inmutable y carga desde el entorno
│   ├── weather.py             # Función externa get_weather (Open-Meteo)
│   ├── rag.py                 # Indexado del PDF y recuperación con citas
│   ├── tools.py               # Herramientas LangChain (@tool) para el LLM
│   └── assistant.py           # Asistente conversacional (chat / stream)
├── data/
│   └── TENERIFE.pdf           # Guía oficial usada como fuente del RAG
├── storage/
│   └── faiss_index/           # Índice FAISS (se genera en la primera ejecución)
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

La app ofrece el chat con el asistente, preguntas de ejemplo, respuesta en
*streaming* y el panel de fuentes citadas en cada respuesta.

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

El resto de parámetros (tamaño de troceado, solapamiento, `top_k` de
recuperación, longitud máxima del historial, rutas del PDF y del índice) se
definen en `core/config.py` mediante la clase `Settings`.

## Índice FAISS

El índice vectorial se **construye en la primera ejecución** a partir de
`data/TENERIFE.pdf` y se persiste en `storage/faiss_index/`. En las ejecuciones
siguientes se carga directamente desde disco, por lo que el arranque es mucho
más rápido. Para forzar su reconstrucción (por ejemplo tras actualizar el PDF o
cambiar el modelo de embeddings), elimina el directorio del índice o invoca
`build_index(force=True)`.

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
- Resumir el historial antiguo en lugar de recortarlo para conservar contexto.
- Soporte multilingüe para turistas que no hablen español.
- Persistencia de conversaciones y caché de respuestas frecuentes.
```
