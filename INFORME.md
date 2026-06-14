# Informe final — Asistente turístico conversacional de Tenerife

Trabajo de fin de máster (Pontia LLM). Asistente conversacional que combina
**RAG** sobre la guía oficial `TENERIFE.pdf` (expuesta como **herramienta**, con
respuestas **ancladas** a los fragmentos recuperados y **fotos** de los lugares),
**diálogo multiturno** con memoria y una **function call** `get_weather`.
Stack: **Google Gemini** vía **LangChain**, vector store **FAISS**, **PyMuPDF**
para las imágenes e interfaz web **Streamlit** con respuesta en *streaming*.

---

## 1. Diseño de la solución

La arquitectura sigue un patrón de **agente con herramientas** (*tool calling*):
el LLM decide en cada turno qué herramienta invocar (o ninguna) y redacta la
respuesta final citando las fuentes. No hay una cadena RAG fija: la recuperación
es una herramienta más que el modelo usa solo cuando la pregunta lo requiere.

### Diagrama del flujo

```
                    ┌──────────────────────────────────────────────┐
   Usuario ───────► │  TouristAssistant (core/assistant.py)        │
  (Streamlit/       │  - SYSTEM_PROMPT (rol + reglas, español)     │
   notebook)        │  - history: memoria multiturno (trim)        │
                    │  - llm.bind_tools([...])                     │
                    └───────────────────┬──────────────────────────┘
                                        │  invoke(history)
                                        ▼
                        ┌───────────────────────────────┐
                        │   Gemini decide: ¿tool call?  │
                        └───────┬───────────────┬───────┘
                       sí (tool_calls)          │ no (respuesta directa)
                                │               │
              ┌─────────────────┴───────────┐   │
              ▼                             ▼   │
   ┌──────────────────────┐   ┌──────────────────────────┐
   │ search_tourist_guide │   │ get_weather(date)        │
   │ (core/tools.py)      │   │ (core/tools.py)          │
   │   └► TouristGuideRAG │   │   └► get_weather()       │
   │      .retrieve()     │   │      (core/weather.py)   │
   │   FAISS similarity   │   │   Open-Meteo (HTTP)      │
   │   top_k fragmentos   │   │   └► fallback simulado   │
   │   + metadatos cita   │   │                          │
   └──────────┬───────────┘   └────────────┬─────────────┘
              │ ToolMessage(context+imgs)   │ ToolMessage(json)
              └───────────────┬─────────────┘
                              ▼
                  ┌───────────────────────────┐
                  │ working += ToolMessage(s) │
                  │ (andamiaje EFÍMERO; no    │
                  │ persiste en la memoria)   │
                  │ bucle hasta respuesta     │
                  │ final (max_tool_rounds)   │
                  └─────────────┬─────────────┘
                                ▼
                  ┌───────────────────────────┐
                  │ Gemini redacta respuesta  │
                  │ final en español, citando │
                  │ páginas de la guía        │
                  └─────────────┬─────────────┘
                                ▼
   Respuesta (stream) + Fuentes (last_sources) + Fotos (last_images)
                                ▼
                              Usuario
```

**Componentes (paquete `core/`):**

- `config.py` — `Settings` inmutable (`@dataclass(frozen=True)`): modelo,
  embeddings, parámetros RAG, rutas y límite de historial. `load_settings`
  lee `.env`.
- `weather.py` — `get_weather`: previsión real de **Open-Meteo** con *fallback*
  determinista y registro en `WEATHER_CALL_LOG`.
- `rag.py` — `TouristGuideRAG`: carga del PDF, troceado, embeddings Gemini,
  índice **FAISS** persistente y recuperación con **citas** e **imágenes** de las
  páginas recuperadas.
- `images.py` — `GuideImageStore`: extrae con **PyMuPDF** las fotos embebidas en
  el PDF, las persiste en `storage/images/` y las indexa por página
  (`images_for_pages`).
- `tools.py` — herramientas `@tool` (`search_tourist_guide`, `get_weather`)
  cuyo JSON Schema se deriva de los *type hints* y *docstrings*.
- `assistant.py` — `TouristAssistant`: memoria, bucle de *tool calling* con
  andamiaje **efímero** (las herramientas no quedan en la memoria), `chat`
  (respuesta completa) y `stream` (tokens), trazas de herramientas.
- `app.py` — interfaz **Streamlit** con *streaming*, panel de fuentes y fotos.

---

## 2. Decisiones técnicas

### Gemini + LangChain
Se usa **Google Gemini** (`gemini-2.5-flash-lite` para generación y
`gemini-embedding-001` para embeddings) por su buen equilibrio coste/latencia y
soporte nativo de *function calling*. **LangChain** aporta la capa de
orquestación común a las sesiones del máster: `init_chat_model`, `bind_tools`,
los tipos de mensaje (`System/Human/AI/ToolMessage`), `trim_messages` y los
*loaders*/*splitters*. Así el código del agente queda agnóstico del proveedor.

### RAG como herramienta, no como cadena fija
Una cadena RAG clásica recupera **siempre** antes de generar. Aquí el asistente
también responde sobre el **tiempo** y mantiene **conversación general**, donde
recuperar del PDF sería inútil o ruidoso. Exponiendo la búsqueda como
**herramienta** (`search_tourist_guide`), el LLM decide cuándo consultar la guía
y cuándo no, puede **reformular la consulta** de recuperación y **combinar**
varias herramientas en un mismo turno (p. ej. recomendar una playa *y* dar el
tiempo de ese día). Es la decisión central del diseño.

### Anclaje al documento (*grounding*) y citas en cada turno
La primera versión presentaba dos problemas observados en pruebas: el modelo
**solo citaba fuentes en el primer turno** y tendía a **responder de memoria** en
las preguntas de seguimiento. La causa era que los mensajes de herramienta (la
petición y el contexto recuperado) **quedaban en el historial**: en los turnos
siguientes el modelo reutilizaba ese contexto antiguo —o su conocimiento
paramétrico— sin volver a invocar la herramienta, de modo que las fuentes (y las
fotos) no se refrescaban. La solución combina dos medidas:

1. **Andamiaje de herramientas efímero**: el bucle de *tool calling* trabaja sobre
   una **copia** del historial; en la memoria persistente solo se guardan los
   turnos de usuario y asistente, no las peticiones ni los resultados de
   herramientas. Así, al no haber contexto recuperado previo, el modelo **vuelve a
   consultar la guía en cada pregunta** y refresca fuentes e imágenes.
2. **Prompt de sistema más estricto**: obliga a usar `search_tourist_guide` para
   **cualquier** pregunta turística (también de seguimiento) y a responder
   **únicamente** con los fragmentos recuperados, sin añadir datos de memoria.

Se conserva así el patrón *RAG como herramienta* (el modelo sigue decidiendo,
reformulando y combinando *tools*) pero con recuperación fiable turno a turno.

### Fotos de la guía (multimodalidad ligera)
La guía incluye una fotografía por lugar. `GuideImageStore` las extrae con
**PyMuPDF** (`extract_image`), descarta las pequeñas (decoraciones) por
`min_image_size` y las mapea por **página** del PDF, derivando un pie de foto del
texto situado justo encima de cada imagen. Como cada fragmento recuperado lleva su
`page`, basta cruzar páginas → fotos para mostrar la imagen del lugar recuperado
junto a la respuesta (hasta `max_images_shown`). Es un enfoque sencillo y robusto:
no requiere un modelo multimodal y reutiliza la propia recuperación textual.

### FAISS vs almacenamiento en memoria
Se elige **FAISS** persistente (`save_local` / `load_local`) frente a un store
en memoria porque el índice se construye **una sola vez** y se **reutiliza**
entre ejecuciones del notebook y reinicios del servidor Streamlit
(`@st.cache_resource`). Evita re-embeddear el PDF en cada arranque (coste y
latencia) y es el estándar de las sesiones de RAG del máster.

### Parámetros de troceado y recuperación
- `chunk_size = 1000`, `chunk_overlap = 150` con
  `RecursiveCharacterTextSplitter`: fragmentos lo bastante grandes para conservar
  contexto de un párrafo/apartado, con solape suficiente para no cortar ideas a
  mitad. `add_start_index=True` para trazabilidad.
- `top_k = 4`: recupera los 4 fragmentos más similares. Suficiente para cubrir la
  respuesta sin saturar el contexto ni diluir la relevancia.
- Cada fragmento lleva metadatos de **cita** (`source_name`, `page`, `chunk_id`),
  que se vuelcan en encabezados `[Fuente i: ..., página N, fragmento M]` y en
  `last_sources` para mostrarlos en la interfaz.

### Control de longitud del historial
El diálogo multiturno crece sin límite, lo que dispara coste y puede degradar la
calidad. Tras cada turno se aplica `trim_messages` (`strategy="last"`,
`include_system=True`, `start_on="human"`) con `max_history_messages = 12`, de
modo que **siempre se conserva el `SystemMessage`** y los turnos más recientes.

### Open-Meteo real + *fallback*
`get_weather` consulta la **API real de Open-Meteo** (sin clave) para una fecha
dada. Si la red o el *parseo* fallan, recurre a una previsión **simulada
determinista** (hash de la fecha → valores plausibles 18–30 °C), de forma que el
asistente **nunca se queda sin respuesta**. La validación de formato
`YYYY-MM-DD` sí se propaga como error controlado, y cada intento se registra en
`WEATHER_CALL_LOG` (`ok`, `source`, `elapsed_s`, `error`) para observabilidad.

---

## 3. Resultados

### Qué demuestra el notebook
El notebook recorre el ciclo completo de extremo a extremo:

1. **Indexado**: construcción del índice FAISS desde `TENERIFE.pdf` y prueba de
   `TouristGuideRAG.search` / `retrieve`, mostrando fragmentos y metadatos.
2. **Herramientas**: invocación aislada de `search_tourist_guide` y
   `get_weather` para ver el JSON Schema que ve el modelo y sus salidas.
3. **Diálogo multiturno**: varias preguntas encadenadas que demuestran memoria
   (la segunda pregunta se apoya en la primera) y el bucle de *tool calling*.
4. **Casos de *tool call***:
   - Pregunta turística → el modelo llama a `search_tourist_guide` y **cita la
     página** de la guía.
   - Pregunta meteorológica (p. ej. *"¿Qué tiempo hará el 2026-06-20?"*) → llama
     a `get_weather` y reporta máx/mín y precipitación.
   - Pregunta mixta → **dos** llamadas en el mismo turno.
   - Pregunta fuera de la guía → el asistente lo dice y **no inventa**.
5. **Trazas**: inspección de `tool_log` (`ToolCallRecord`) y `WEATHER_CALL_LOG`.

### Cómo se citan las fuentes y se muestran las fotos
La recuperación adjunta a cada fragmento su `source_name`, `page` y `chunk_id`.
`format_context` los inserta como encabezados visibles para el modelo, el
`SYSTEM_PROMPT` le **obliga a citar la página**, y la interfaz muestra las
fuentes (`render_sources`) en un desplegable con *snippet* de cada fragmento.
Además, a partir de la `page` de los fragmentos recuperados se obtienen las
**fotos** de esas páginas (`GuideImageStore.images_for_pages`), que la interfaz
muestra con `render_images`. Como el andamiaje de *tool calling* es efímero, la
guía se consulta **en cada turno** y, por tanto, fuentes y fotos se refrescan
siempre (no solo en la primera pregunta).

---

## 4. Evaluación

Al no disponer de un *dataset* de referencia anotado, la evaluación es
**heurística y reproducible**, sobre un pequeño conjunto de preguntas de prueba:

- **Cobertura de herramienta (*tool routing*)**: se comprueba que cada tipo de
  pregunta dispara la herramienta esperada (turística → `search_tourist_guide`,
  meteorológica → `get_weather`, mixta → ambas), inspeccionando `tool_log`.
- **Fidelidad / *groundedness***: para preguntas sobre la guía se verifica que la
  respuesta **cita página** y que el contenido procede de los fragmentos
  recuperados (control anti-alucinación; las preguntas fuera de la guía deben
  responderse con un "no aparece en la guía").
- **Relevancia de recuperación**: revisión manual de los `top_k` fragmentos para
  confirmar que contienen la información buscada.
- **Robustez de `get_weather`**: prueba del *fallback* (forzando fallo de red) y
  de la validación de fecha; se mide `elapsed_s` por llamada.
- **Latencia / observabilidad**: tiempos por herramienta vía `ToolCallRecord` y
  `WEATHER_CALL_LOG`, agregables con `pandas`/`matplotlib`.

---

## 5. Limitaciones

- **Conocimiento acotado al PDF**: el RAG solo sabe lo que contiene
  `TENERIFE.pdf`; preguntas fuera de su alcance se responden con cautela pero no
  con datos externos.
- **Evaluación sin *ground truth***: las métricas son heurísticas y parcialmente
  manuales; no hay puntuaciones automáticas tipo *RAGAS*.
- **Meteorología agregada**: `get_weather` da previsión **diaria** (máx/mín y
  lluvia) para un punto fijo (Santa Cruz), sin granularidad horaria ni por zonas
  de la isla; el *fallback* simulado no es una previsión real.
- **`top_k` y troceado fijos**: no hay *reranking* ni recuperación adaptativa.
- **Memoria por recorte simple**: `trim_messages` descarta turnos antiguos sin
  resumirlos, por lo que se pierde contexto lejano en charlas largas.
- **Fotos asociadas por página**: la imagen se vincula a la página del fragmento,
  no al lugar concreto. Si una página mezcla varios sitios, puede mostrarse una
  foto de un lugar contiguo; el pie de foto es una heurística del texto superior.
- **Coste de recuperar en cada turno**: forzar la consulta a la guía en todos los
  turnos turísticos mejora el anclaje y las citas, a cambio de una llamada de
  recuperación adicional por turno.
- **Sin persistencia de conversación** entre sesiones de servidor.

---

## 6. Mejoras futuras

**Ya implementado (bonus del enunciado):**
- *Streaming* real de tokens (`TouristAssistant.stream` + `st.write_stream`).
- **Despliegue web** con Streamlit (chat, panel de fuentes, parámetros y
  reinicio de conversación).
- **Anclaje al documento** (*grounding*) con citas y recuperación en cada turno.
- **Fotos de los lugares** del PDF mostradas junto a la respuesta (`core/images.py`).
- Observabilidad básica (trazas de herramientas y log de meteorología).

**Pendiente / futuro:**
- **Agentes multi-herramienta**: ampliar el catálogo (mapas/rutas, eventos,
  reservas) y planificación más sofisticada.
- **Multimodalidad avanzada**: aprovechar Gemini para *entender* las imágenes
  (reconocer un lugar de una foto, leer carteles) y mejorar el emparejamiento
  foto-lugar más allá de la página.
- **Observabilidad avanzada**: trazas distribuidas (p. ej. LangSmith), métricas
  de coste/tokens y *dashboards*.
- **Evaluación automática**: *RAGAS* u otro *framework* con *ground truth*.
- **Memoria con resumen** y persistencia de conversaciones.
- **Recuperación mejorada**: *reranking*, *hybrid search* y `top_k` adaptativo.

---

## 7. Trazabilidad: requisitos mínimos → dónde se cumplen

| Requisito mínimo del enunciado | Dónde se cumple |
|---|---|
| **RAG sobre `TENERIFE.pdf`** | `core/rag.py` (`TouristGuideRAG.build_index`, `search`, `retrieve`); PDF en `data/TENERIFE.pdf` |
| **PDF expuesto como HERRAMIENTA** | `core/tools.py` → `@tool search_tourist_guide` |
| **Vector store FAISS** | `core/rag.py` (`FAISS.from_documents`, `save_local`/`load_local`); índice en `storage/faiss_index` |
| **Embeddings** | `core/rag.py` → `GoogleGenerativeAIEmbeddings(models/gemini-embedding-001)` |
| **Troceado del documento** | `core/rag.py` → `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)` |
| **Citado de fuentes** | `core/rag.py` (`format_context`, `_doc_to_source`, `last_sources`); `app.py` (`render_sources`) |
| **Anclaje al documento (*grounding*) y citas en cada turno** | `core/assistant.py` → `SYSTEM_PROMPT` + andamiaje efímero en `chat`/`stream`/`_run_tool_rounds` |
| **Imágenes del PDF mostradas en el chat** | `core/images.py` (`GuideImageStore`); `core/rag.py` (`last_images`); `app.py` (`render_images`) |
| **Function call `get_weather`** | `core/weather.py` (`get_weather`) + `core/tools.py` (`@tool get_weather`) |
| **API externa real** | `core/weather.py` → Open-Meteo (`OPEN_METEO_URL`) con *fallback* |
| **Diálogo multiturno con memoria** | `core/assistant.py` → `self.history`, `_trim_history` |
| **Tool calling con Gemini + LangChain** | `core/assistant.py` → `init_chat_model` + `bind_tools` + `_run_tool_rounds` |
| **Prompt de sistema (rol del asistente)** | `core/assistant.py` → `SYSTEM_PROMPT` (español) |
| **Configuración por entorno** | `core/config.py` → `Settings`, `load_settings`, `.env` |
| **Notebook de demostración** | `notebook` (indexado, herramientas, diálogo, *tool calls*, evaluación) |
| **Control de parámetros del modelo** | `core/config.py` (`temperature`, `top_p`, `max_output_tokens`); panel en `app.py` |
| **Bonus: streaming** | `core/assistant.py` → `stream`; `app.py` → `st.write_stream` |
| **Bonus: despliegue web** | `app.py` (Streamlit) |
| **Observabilidad** | `ToolCallRecord` / `tool_log` (`assistant.py`), `WEATHER_CALL_LOG` (`weather.py`) |
