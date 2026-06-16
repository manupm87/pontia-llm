# Informe final — Asistente turístico conversacional de Tenerife

Trabajo de fin de módulo LLM (Pontia). Asistente conversacional que combina
**RAG** sobre la guía oficial `TENERIFE.pdf` (expuesta como **herramienta**, con
respuestas **ancladas** a los fragmentos recuperados y **fotos** de los lugares
intercaladas en la respuesta), **diálogo multiturno** con memoria y **function
calls** externas: `get_weather` (tiempo), `get_sea_conditions` (estado del mar) y
`resolve_date` (fechas relativas → ISO). Incorpora además **guardarraíles**
(detección de manipulación de instrucciones y filtros LLM opcionales),
**razonamiento en streaming** y un **harness de evaluación** con LLM-as-judge.
Stack: **Google Gemini** vía **LangChain**, vector store **FAISS**, **PyMuPDF**
para las imágenes e interfaz web **Streamlit** con respuesta en *streaming*.

🚀 **Demo en vivo:** la aplicación está desplegada en Streamlit Community Cloud en
<https://pontia-llm-zrvfwbe9t6m28chhrn6cms.streamlit.app/>.

---

## 1. Diseño de la solución

La arquitectura sigue un patrón de **agente con herramientas** (*tool calling*):
el LLM decide en cada turno qué herramienta invocar (o ninguna) y redacta la
respuesta final apoyándose en lo recuperado. No hay una cadena RAG fija: la
recuperación es una herramienta más que el modelo usa solo cuando la pregunta lo
requiere. Antes de gastar en recuperación o generación, un **guardarraíl de
entrada** filtra los intentos de manipulación.

### Diagrama del flujo

```
                    ┌──────────────────────────────────────────────┐
   Usuario ───────► │  Guardarraíl de entrada (core/guardrails.py) │
  (Streamlit/       │  - reglas anti-inyección (siempre)           │
   notebook)        │  - clasificador de tema LLM (opcional)       │
                    └───────────────────┬──────────────────────────┘
                              bloqueado  │  permitido
                            ◄─ rechazo ──┤
                                         ▼
                    ┌──────────────────────────────────────────────┐
                    │  TouristAssistant (core/assistant.py)        │
                    │  - SYSTEM_PROMPT (rol + reglas, español)     │
                    │  - history: memoria multiturno (trim)        │
                    │  - llm.bind_tools([...])                     │
                    └───────────────────┬──────────────────────────┘
                                        │  invoke(working)
                                        ▼
                        ┌───────────────────────────────┐
                        │   Gemini decide: ¿tool call?  │
                        └───────┬───────────────┬───────┘
                       sí (tool_calls)          │ no (respuesta directa)
                                │               │
        ┌───────────────┬───────┴───────┬───────────────┐
        ▼               ▼               ▼               ▼
 ┌────────────┐  ┌────────────┐  ┌──────────────┐  ┌────────────┐
 │ search_    │  │ get_weather│  │ get_sea_     │  │ resolve_   │
 │ tourist_   │  │ (date)     │  │ conditions   │  │ date(expr) │
 │ guide      │  │            │  │ (date)       │  │            │
 │ └►RAG.     │  │ └►weather  │  │ └►sea        │  │ └►dates    │
 │   retrieve │  │   Open-    │  │   Open-Meteo │  │   ES→ISO   │
 │ FAISS top_k│  │   Meteo    │  │   Marine     │  │            │
 │ +citas+img │  │ +fallback  │  │ +fallback    │  │            │
 └─────┬──────┘  └─────┬──────┘  └──────┬───────┘  └─────┬──────┘
       │ artifact       │ json          │ json           │ ISO
       └────────────────┴───────┬───────┴────────────────┘
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
                  │ final en español (+ razo- │
                  │ namiento en streaming)    │
                  └─────────────┬─────────────┘
                                ▼
   Respuesta (stream) + Fotos intercaladas + Fuentes (last_sources)
                                ▼
                              Usuario
```

**Componentes (paquete `core/`):**

- `config.py` — `Settings` inmutable (`@dataclass(frozen=True)`): modelo,
  embeddings, parámetros RAG, rutas, `thinking_budget` y límite de historial.
  `load_settings` lee `.env` (tolera valores numéricos malformados sin abortar).
- `rag.py` — `TouristGuideRAG`: carga del PDF, troceado, embeddings Gemini,
  índice **FAISS** persistente y recuperación con **citas** e **imágenes** de las
  páginas recuperadas.
- `images.py` — `GuideImageStore`: extrae con **PyMuPDF** las fotos embebidas en
  el PDF, las persiste en `storage/images/` (con `manifest.json`) y las indexa por
  página (`images_for_pages`).
- `photo_match.py` — lógica pura que **intercala** cada foto junto a la primera
  línea de la respuesta que menciona su lugar (`plan_inline_images`).
- `weather.py` — `get_weather`: previsión real de **Open-Meteo** con *fallback*
  determinista y registro en `WEATHER_CALL_LOG`.
- `sea.py` — `get_sea_conditions`: oleaje (altura y periodo) de **Open-Meteo
  Marine** con *fallback* y `SEA_CALL_LOG`; replica el patrón de `weather.py`.
- `meteo.py` — `fetch_with_fallback`: orquestación común a `weather`/`sea`
  (validar fecha, llamar a la API, cronometrar, registrar y caer al simulador).
- `dates.py` — `resolve_date`: expresiones relativas en español ("mañana", "este
  finde", "el lunes que viene", "en 3 días") → ISO `YYYY-MM-DD`.
- `text.py` — `normalize_text`: normalización compartida (sin acentos ni
  mayúsculas) para comparaciones laxas en guardarraíles, fechas, evaluación y fotos.
- `tools.py` — herramientas `@tool` (`search_tourist_guide`, `get_weather`,
  `get_sea_conditions`, `resolve_date`) cuyo JSON Schema se deriva de los *type
  hints* y *docstrings*.
- `guardrails.py` — `Guardrails`: detección de inyección por reglas (siempre) y,
  opcionalmente, clasificador de tema y juez de fidelidad con LLM.
- `evaluation.py` — *harness* de evaluación: casos, métricas, LLM-as-judge y
  reporte (`pandas`/`matplotlib`).
- `assistant.py` — `TouristAssistant`: memoria, bucle de *tool calling* con
  andamiaje **efímero**, `prepare`/`chat`/`stream`/`stream_reasoning_and_answer`,
  guardarraíles y observabilidad de tokens.
- `app.py` + `ui_theme.py` — interfaz **Streamlit** con *streaming*, razonamiento
  en vivo, fotos intercaladas, panel de fuentes, controles y coste; tema visual.

---

## 2. Decisiones técnicas

### Gemini + LangChain
Se usa **Google Gemini** (`gemini-2.5-flash-lite` para generación y
`gemini-embedding-001` para embeddings) por su buen equilibrio coste/latencia y
soporte nativo de *function calling*. **LangChain** aporta la capa de
orquestación común a las sesiones del módulo: `init_chat_model`, `bind_tools`,
los tipos de mensaje (`System/Human/AI/ToolMessage`), `trim_messages` y los
*loaders*/*splitters*. Así el código del agente queda agnóstico del proveedor.

### RAG como herramienta, no como cadena fija
Una cadena RAG clásica recupera **siempre** antes de generar. Aquí el asistente
también responde sobre el **tiempo** y el **estado del mar**, resuelve **fechas**
y mantiene **conversación general**, donde recuperar del PDF sería inútil o
ruidoso. Exponiendo la búsqueda como **herramienta** (`search_tourist_guide`), el
LLM decide cuándo consultar la guía y cuándo no, puede **reformular la consulta**
de recuperación y **combinar** varias herramientas en un mismo turno (p. ej.
resolver la fecha de "este finde", recomendar una playa *y* dar el tiempo y el
estado del mar de ese día). Es la decisión central del diseño.

### Anclaje al documento (*grounding*) y citas en cada turno
La primera versión presentaba dos problemas observados en pruebas: el modelo
**solo citaba fuentes en el primer turno** y tendía a **responder de memoria** en
las preguntas de seguimiento. La causa era que los mensajes de herramienta (la
petición y el contexto recuperado) **quedaban en el historial**: en los turnos
siguientes el modelo reutilizaba ese contexto antiguo —o su conocimiento
paramétrico— sin volver a invocar la herramienta, de modo que las fuentes (y las
fotos) no se refrescaban. La solución combina dos medidas:

1. **Andamiaje de herramientas efímero**: el bucle de *tool calling* (`prepare`)
   trabaja sobre una **copia** del historial; en la memoria persistente solo se
   guardan los turnos de usuario y asistente, no las peticiones ni los resultados
   de herramientas. Así, al no haber contexto recuperado previo, el modelo
   **vuelve a consultar la guía en cada pregunta** y refresca fuentes e imágenes.
2. **Prompt de sistema más estricto**: obliga a usar `search_tourist_guide` para
   **cualquier** pregunta turística (también de seguimiento) y a responder
   **únicamente** con los fragmentos recuperados, sin añadir datos de memoria.

Se conserva así el patrón *RAG como herramienta* (el modelo sigue decidiendo,
reformulando y combinando *tools*) pero con recuperación fiable turno a turno.

### Fotos de la guía intercaladas (multimodalidad ligera)
La guía incluye una fotografía por lugar. `GuideImageStore` las extrae con
**PyMuPDF** (`extract_image`), descarta las pequeñas (decoraciones) por
`min_image_size` y las mapea por **página** del PDF, derivando un pie de foto del
texto situado justo encima de cada imagen. Como cada fragmento recuperado lleva su
`page`, basta cruzar páginas → fotos para obtener la imagen del lugar recuperado.
En lugar de amontonarlas al final, `photo_match.plan_inline_images` **intercala**
cada foto junto a la primera línea de la respuesta que nombra su lugar (cruzando
los tokens distintivos del pie con las palabras de la línea), y agrupa al final
las no mencionadas (hasta `max_images_shown`). Es un enfoque sencillo y robusto:
no requiere un modelo multimodal y reutiliza la propia recuperación textual.

### Function calls: tiempo, mar y fechas
- `get_weather` y `get_sea_conditions` consultan las **APIs reales de Open-Meteo**
  (previsión y marina, sin clave) para una fecha dada. Si la red o el *parseo*
  fallan, recurren a una previsión **simulada determinista** (hash de la fecha →
  valores plausibles), de forma que el asistente **nunca se queda sin respuesta**.
  La validación de formato `YYYY-MM-DD` sí se propaga como error controlado, y
  cada intento se registra en `WEATHER_CALL_LOG`/`SEA_CALL_LOG` (`ok`, `source`,
  `elapsed_s`, `error`) para observabilidad. El flujo común (validar, llamar,
  cronometrar, registrar y caer al simulador) se factoriza en `meteo.py`
  (`fetch_with_fallback`) para no duplicarlo; cada módulo solo aporta su URL,
  parámetros, *parser* y simulador.
- `resolve_date` traduce expresiones relativas en español a ISO. El prompt obliga
  a **resolver la fecha primero** y pasar el `YYYY-MM-DD` a `get_weather`/
  `get_sea_conditions`, de modo que "¿qué tiempo hará este finde?" funciona sin que
  el modelo tenga que calcular fechas (tarea en la que los LLM fallan a menudo). El
  prompt de sistema se ancla además a la fecha actual del servidor.

### Guardarraíles (defensa en profundidad)
Inspirados en la sesión 05 del módulo, añaden una capa de validación
**independiente del prompt** (`core/guardrails.py`):

- **Entrada (siempre activa, gratis)**: `detect_injection` aplica reglas rápidas
  (regex sobre texto normalizado) para detectar intentos de manipular las
  instrucciones ("ignora tus instrucciones", "muéstrame el system prompt",
  *jailbreak*, volcado literal del documento). Si saltan, el turno se rechaza
  **antes** de gastar en RAG o generación. Los patrones acotan el hueco entre
  disparadores para evitar falsos positivos y ReDoS.
- **Avanzados (opcionales, con LLM)**: `build_llm_guardrails` añade un
  **clasificador de tema** (rechaza lo ajeno al turismo de Tenerife) y un **juez de
  fidelidad** (*grounding*) vía `with_structured_output`. Consumen tokens, por lo
  que se activan desde la barra lateral. La lógica de reglas es pura y testeable;
  las comprobaciones con LLM se **inyectan** y *fallan en abierto* (ante un error
  del juez, no se bloquea al usuario).

### Razonamiento en *streaming*
Si el modelo lo admite (`thinking_budget != 0`), se activa el "thinking" de Gemini
con resúmenes de razonamiento. `assistant.stream_reasoning_and_answer` separa cada
*chunk* en `(is_thought, text)` (`_split_content` tolera las distintas formas del
SDK) y emite el razonamiento y la respuesta por separado: la interfaz muestra el
razonamiento en vivo y luego lo colapsa, mientras **solo la respuesta** se persiste
en el historial. Los nombres de los parámetros de "thinking" varían entre versiones
del SDK, así que el modelo se construye probando combinaciones en cascada y, como
último recurso, sin razonamiento, para no romper el arranque.

### FAISS vs almacenamiento en memoria
Se elige **FAISS** persistente (`save_local` / `load_local`) frente a un store
en memoria porque el índice se construye **una sola vez** y se **reutiliza**
entre ejecuciones del notebook y reinicios del servidor Streamlit
(`@st.cache_resource`). Evita re-embeddear el PDF en cada arranque (coste y
latencia) y es el estándar de las sesiones de RAG del módulo.

### Parámetros de troceado y recuperación
- `chunk_size = 500`, `chunk_overlap = 100` con
  `RecursiveCharacterTextSplitter`: fragmentos lo bastante grandes para conservar
  contexto de un párrafo/apartado, con solape suficiente para no cortar ideas a
  mitad. `add_start_index=True` para trazabilidad.
- `top_k = 5`: recupera los 5 fragmentos más similares. Suficiente para cubrir la
  respuesta sin saturar el contexto ni diluir la relevancia.
- Cada fragmento lleva metadatos de **cita** (`source_name`, `page`, `chunk_id`),
  que se vuelcan en encabezados `[Fuente i: ..., página N, fragmento M]` y en el
  *artifact* de cada llamada a la herramienta para mostrarlos en la interfaz.

### Control de longitud del historial
El diálogo multiturno crece sin límite, lo que dispara coste y puede degradar la
calidad. Tras cada turno se aplica `trim_messages` (`strategy="last"`,
`include_system=True`, `start_on="human"`) con `max_history_messages = 12`, de
modo que **siempre se conserva el `SystemMessage`** y los turnos más recientes.

---

## 3. Resultados

### Qué demuestra el notebook
El **[notebook de demostración](notebook_asistente_tenerife.ipynb)**
(`notebook_asistente_tenerife.ipynb`) recorre el ciclo completo de extremo a
extremo, con una sección por cada capacidad en orden natural del pipeline:

1. **Indexado**: construcción del índice FAISS desde `TENERIFE.pdf` y prueba de
   `TouristGuideRAG.search` / `retrieve`, mostrando fragmentos y metadatos.
2. **Herramientas**: invocación aislada de `search_tourist_guide`, `get_weather`,
   `get_sea_conditions` y `resolve_date` para ver el JSON Schema que ve el modelo
   y sus salidas.
3. **Diálogo multiturno**: varias preguntas encadenadas que demuestran memoria
   (la segunda pregunta se apoya en la primera) y el bucle de *tool calling*.
4. **Casos de *tool call***:
   - Pregunta turística → el modelo llama a `search_tourist_guide` y responde
     **solo** con lo recuperado (las fuentes se muestran aparte).
   - Pregunta meteorológica con fecha relativa (p. ej. *"¿Qué tiempo hará este
     finde?"*) → `resolve_date` y luego `get_weather`.
   - Pregunta sobre el mar → `get_sea_conditions` (oleaje, baño, surf).
   - Pregunta mixta → **varias** llamadas en el mismo turno.
   - Pregunta fuera de la guía → el asistente lo dice y **no inventa**.
5. **Trazas**: inspección de `tool_log` (`ToolCallRecord`), `WEATHER_CALL_LOG` y
   `SEA_CALL_LOG`, y del uso de tokens (`total_usage` + coste estimado).
6. **Evaluación**: ejecución del *harness* (sección 4) con métricas y gráfico.

### Cómo se anclan las respuestas y se muestran las fotos
La recuperación adjunta a cada fragmento su `source_name`, `page` y `chunk_id`.
`format_context` los inserta como encabezados visibles para el modelo y el
`SYSTEM_PROMPT` le **obliga a responder solo con lo recuperado**; la interfaz
muestra las fuentes (`render_sources`) en un desplegable con *snippet* de cada
fragmento. Además, a partir de la `page` de los fragmentos recuperados se obtienen
las **fotos** de esas páginas (`GuideImageStore.images_for_pages`), que la interfaz
**intercala** en la respuesta junto al lugar mencionado (`plan_inline_images` +
`render_images`). Como el andamiaje de *tool calling* es efímero, la guía se
consulta **en cada turno** y, por tanto, fuentes y fotos se refrescan siempre (no
solo en la primera pregunta).

---

## 4. Evaluación

Al no disponer de un *dataset* de referencia anotado, la evaluación es
**heurística y reproducible**. Se implementa como un *harness* en
`core/evaluation.py` (inspirado en las sesiones 04/06) y se ejecuta de extremo a
extremo con `python -m scripts.run_eval`, que escribe `storage/eval_results.csv`
(tabla por caso) y `storage/eval_summary.png` (gráfico de métricas).

**Conjunto de casos** (`default_dataset`): preguntas **dentro de ámbito**
(playas, Teide, La Laguna, gastronomía, Anaga) con palabras clave esperadas, y
**fuera de ámbito** (deporte, código, intento de *jailbreak*) que deben
rechazarse. Cada `EvalCase` declara su `kind` y sus `expected_keywords`.

**Métricas** (`summarize`):

- **Acierto de recuperación (`retrieval_hit_rate`)**: el contexto recuperado
  contiene alguna palabra clave esperada (comparación normalizada con `text.py`).
- **Fidelidad / *groundedness* (`faithfulness_rate`)**: un **juez LLM** opcional
  (`build_llm_guardrails`) decide si la respuesta se apoya solo en el contexto.
  La puntuación es *honesta*: sin juez, un caso correcto no garantiza fidelidad
  (`score_correct` lo documenta explícitamente).
- **Tasa de rechazo fuera de ámbito (`refusal_rate_out_of_scope`)**: los casos
  ajenos al turismo deben responderse con un rechazo (`is_refusal`).
- ***Accuracy* agregada**: combina, por tipo de caso, recuperación, no-rechazo y
  fidelidad (cuando se juzgó).
- **Cobertura de herramienta (*tool routing*)**: `EvalResult.tool_names` registra
  qué herramientas disparó cada pregunta, para verificar el enrutado esperado.

La lógica de puntuación es **pura y testeable** (cubierta en `tests/`); `pandas`
y `matplotlib` se importan de forma perezosa. Complementan a este *harness* la
**robustez de las APIs** (prueba del *fallback* forzando fallo de red y de la
validación de fecha, midiendo `elapsed_s`) y la **observabilidad** de latencias
por herramienta vía `ToolCallRecord` y los *call logs*.

---

## 5. Limitaciones

- **Conocimiento acotado al PDF**: el RAG solo sabe lo que contiene
  `TENERIFE.pdf`; preguntas fuera de su alcance se responden con cautela pero no
  con datos externos.
- **Evaluación sin *ground truth***: las métricas son heurísticas (palabras clave
  + juez LLM); no hay puntuaciones tipo *RAGAS* con referencias anotadas.
- **Meteorología y mar agregados**: `get_weather`/`get_sea_conditions` dan
  previsión **diaria** para un punto fijo (Santa Cruz), sin granularidad horaria
  ni por zonas; el *fallback* simulado no es una previsión real.
- **`top_k` y troceado fijos**: no hay *reranking* ni recuperación adaptativa.
- **Memoria por recorte simple**: `trim_messages` descarta turnos antiguos sin
  resumirlos, por lo que se pierde contexto lejano en charlas largas.
- **Fotos asociadas por página**: la imagen se vincula a la página del fragmento,
  no al lugar concreto. Si una página mezcla varios sitios, puede mostrarse una
  foto de un lugar contiguo; el pie de foto es una heurística del texto superior.
- **Guardarraíles por reglas**: la detección de inyección es por patrones (puede
  evadirse con redacciones nuevas); los filtros LLM más robustos son opcionales
  por su coste en tokens.
- **Coste de recuperar en cada turno**: forzar la consulta a la guía en todos los
  turnos turísticos mejora el anclaje, a cambio de una llamada de recuperación
  adicional por turno; el *streaming* añade otra generación.
- **Sin persistencia de conversación** entre sesiones de servidor.

---

## 6. Mejoras futuras

**Ya implementado (incluye bonus del enunciado):**
- *Streaming* real de tokens y **razonamiento en vivo** ("thinking" de Gemini).
- **Despliegue web** con Streamlit (chat, fotos intercaladas, panel de fuentes,
  parámetros, controles de streaming/razonamiento/guardarraíles y reinicio).
- **Anclaje al documento** (*grounding*) con recuperación en cada turno.
- **Fotos de los lugares** del PDF **intercaladas** junto a su mención.
- **Más function calls**: estado del mar (`get_sea_conditions`) y resolución de
  fechas relativas (`resolve_date`), además del tiempo.
- **Guardarraíles** de entrada (anti-inyección, siempre) y filtros LLM opcionales.
- **Harness de evaluación** con LLM-as-judge, métricas y reporte (`run_eval`).
- Observabilidad de tokens/coste y trazas de herramientas y de meteorología/mar.

**Pendiente / futuro:**
- **Agentes multi-herramienta**: ampliar el catálogo (mapas/rutas, eventos,
  reservas) y planificación más sofisticada.
- **Multimodalidad avanzada**: aprovechar Gemini para *entender* las imágenes
  (reconocer un lugar de una foto, leer carteles) y mejorar el emparejamiento
  foto-lugar más allá de la página.
- **Observabilidad avanzada**: trazas distribuidas (p. ej. LangSmith) y
  *dashboards* de coste/tokens.
- **Evaluación automática**: *RAGAS* u otro *framework* con *ground truth*.
- **Memoria con resumen** y persistencia de conversaciones.
- **Recuperación mejorada**: *reranking*, *hybrid search* y `top_k` adaptativo.
- **Soporte multilingüe** para turistas que no hablen español.

---

## 7. Trazabilidad: requisitos mínimos → dónde se cumplen

| Requisito mínimo del enunciado | Dónde se cumple |
|---|---|
| **RAG sobre `TENERIFE.pdf`** | `core/rag.py` (`TouristGuideRAG.build_index`, `search`, `retrieve`); PDF en `data/TENERIFE.pdf` |
| **PDF expuesto como HERRAMIENTA** | `core/tools.py` → `@tool search_tourist_guide` |
| **Vector store FAISS** | `core/rag.py` (`FAISS.from_documents`, `save_local`/`load_local`); índice en `storage/faiss_index` |
| **Embeddings** | `core/rag.py` → `GoogleGenerativeAIEmbeddings(models/gemini-embedding-001)` |
| **Troceado del documento** | `core/rag.py` → `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)` |
| **Citado de fuentes** | `core/rag.py` (`format_context`, `_doc_to_source`); citas por llamada en el *artifact* de la herramienta; `app.py` (`render_sources`) |
| **Anclaje al documento (*grounding*) y citas en cada turno** | `core/assistant.py` → `SYSTEM_PROMPT` + andamiaje efímero en `prepare`/`chat`/`stream` |
| **Imágenes del PDF mostradas (intercaladas) en el chat** | `core/images.py` (`GuideImageStore`); `core/photo_match.py` (`plan_inline_images`); `app.py` (`render_images`) |
| **Function call `get_weather`** | `core/weather.py` (`get_weather`) + `core/tools.py` (`@tool get_weather`) |
| **Function call `get_sea_conditions`** | `core/sea.py` (`get_sea_conditions`) + `core/tools.py` (`@tool get_sea_conditions`) |
| **Function call `resolve_date`** | `core/dates.py` (`resolve_date`) + `core/tools.py` (`@tool resolve_date`) |
| **API externa real** | `core/weather.py` / `core/sea.py` → Open-Meteo (`meteo.fetch_with_fallback`) con *fallback* |
| **Diálogo multiturno con memoria** | `core/assistant.py` → `self.history`, `_trim_history` |
| **Tool calling con Gemini + LangChain** | `core/assistant.py` → `init_chat_model` + `bind_tools` + bucle en `prepare`/`_execute_tool_call` |
| **Prompt de sistema (rol del asistente)** | `core/assistant.py` → `build_system_prompt` (español, anclado a la fecha) |
| **Guardarraíles (entrada/salida)** | `core/guardrails.py` (`Guardrails`, `detect_injection`, `build_llm_guardrails`); barra lateral en `app.py` |
| **Configuración por entorno** | `core/config.py` → `Settings`, `load_settings`, `.env` |
| **Notebook de demostración** | [`notebook_asistente_tenerife.ipynb`](notebook_asistente_tenerife.ipynb) (13 secciones: indexado, fotos, herramientas, diálogo, streaming, guardarraíles, evaluación, observabilidad) |
| **Control de parámetros del modelo** | `core/config.py` (`temperature`, `top_p`, `max_output_tokens`, `thinking_budget`); panel en `app.py` |
| **Bonus: streaming + razonamiento** | `core/assistant.py` → `stream`/`stream_reasoning_and_answer`; `app.py` → `st.write_stream` |
| **Bonus: despliegue web** | `app.py` + `ui_theme.py` (Streamlit); desplegado en vivo: <https://pontia-llm-zrvfwbe9t6m28chhrn6cms.streamlit.app/> |
| **Bonus: evaluación** | `core/evaluation.py` (`run_evaluation`, `summarize`); `scripts/run_eval.py` |
| **Observabilidad** | `ToolCallRecord` / `tool_log` (`assistant.py`), `WEATHER_CALL_LOG` (`weather.py`), `SEA_CALL_LOG` (`sea.py`), `total_usage` + `estimate_cost` |
