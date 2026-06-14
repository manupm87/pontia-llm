# Mesa de ayuda interna con RAG

Aplicación Streamlit que materializa el caso de uso de RAG visto en la sesión 3 como una **mesa de ayuda interna de primer nivel**. El objetivo no es exponer controles técnicos sin más, sino simular cómo un equipo de soporte resolvería solicitudes de empleados usando una base documental corporativa.

La app recupera evidencias de una carpeta de PDFs corporativos, redacta una respuesta operativa y muestra fuentes auditables. También incluye una zona para inspeccionar el retrieval y una evaluación del pipeline completo.

## Qué hace

- Carga todos los PDFs de `../inputs/company_docs` y los trocea con `RecursiveCharacterTextSplitter`.
- Indexa los fragmentos en una `InMemoryVectorStore` usando embeddings de Google Gemini.
- Presenta casos reales de soporte interno: accesos, correo, vacaciones, seguridad, gastos y operaciones de cliente.
- Recupera los chunks más relevantes para cada solicitud con filtros opcionales por área (IT, RRHH, Seguridad, General).
- Genera respuestas en streaming con Gemini en formato de resolución: `Resolución`, `Pasos` y `Cuándo escalar`.
- Muestra las fuentes utilizadas (documento, página y chunk) como chips trazables.
- Permite clasificación automática y reescritura de la consulta (query analysis con `with_structured_output`).
- Incluye un inspector de evidencias para ver qué recupera el sistema antes de generar.
- Incluye una evaluación que ejecuta recuperación, generación y rechazo para preguntas de control.
- Permite reindexar con nuevos parámetros de chunking y recuperación.
- Mantiene historial conversacional, exportable a Markdown.

Si el contexto no resuelve la pregunta, el asistente lo reconoce literalmente: "No tengo información suficiente en la base documental".

## Pantallas principales

- **Resolver solicitudes:** bandeja con casos habituales y entrada libre para describir una solicitud de empleado.
- **Auditar recuperación:** laboratorio de retrieval para inspeccionar evidencias, query reescrita y categoría elegida.
- **Evaluar RAG:** pruebas de control para comprobar recuperación, generación y rechazo sin contexto suficiente.

## API keys

```bash
cp .env.template .env
```

Rellena al menos `GOOGLE_API_KEY`. Las demás variables son opcionales y permiten cambiar modelo, defaults de retrieval y ruta de la base documental.

```text
GOOGLE_API_KEY="..."
RAG_GENERATION_MODEL="gemini-2.5-flash-lite"
RAG_EMBEDDING_MODEL="models/gemini-embedding-001"
RAG_DEFAULT_SOURCE_PATH="../inputs/company_docs"
RAG_DEFAULT_TOP_K="3"
RAG_DEFAULT_CHUNK_SIZE="500"
RAG_DEFAULT_CHUNK_OVERLAP="80"
RAG_TIMEOUT_SECONDS="45"
```

## macOS y Linux

```bash
cd sesion_03/rag_assistant
make setup
make run
```

## Windows

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.template .env
streamlit run app.py
```

## Estructura

```
rag_assistant/
├── app.py                  # Orquestación de la app Streamlit
├── core/
│   ├── config.py           # Carga de .env y Settings inmutable
│   ├── categories.py       # Categorías heurísticas (IT/RRHH/Seguridad/General)
│   ├── models.py           # Dataclasses inmutables del dominio
│   ├── rag_engine.py       # Carga, chunking, embeddings, retrieval, generación
│   ├── state.py            # Estado de sesión Streamlit
│   ├── ui.py               # Estilos, layout y componentes visuales
│   └── use_cases.py        # Casos sugeridos y evaluación básica
├── .streamlit/config.toml  # Tema y paleta
├── Makefile
├── requirements.txt
├── .env.template
└── README.md
```

## Notas de diseño

- `RagEngine` aísla LangChain de Streamlit. El motor expone `build_index_from_path`, `build_index_from_bytes`, `analyze_query`, `retrieve` y `stream_answer`.
- El índice se devuelve como un `IndexHandle` inmutable con su `IndexFingerprint`, lo que facilita una futura persistencia o caché por contenido.
- La generación es siempre en streaming. El prompt está orientado a soporte interno: respuesta accionable, pasos y criterio de escalado cuando exista en el contexto.
- Las citas se construyen a partir de los chunks recuperados; no se le pide al modelo que las inserte, así evitamos fuentes alucinadas.
- La heurística de categorías simula metadatos de área. En producción se sustituiría por metadatos reales del corpus.
- El corpus de ejemplo se genera con `../scripts/generate_company_docs.py` a partir de Markdown versionado en `../inputs/company_docs_src`.
