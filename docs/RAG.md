# El proceso RAG del asistente de Tenerife

Este documento explica cómo funciona la recuperación aumentada por generación
(*Retrieval-Augmented Generation*, RAG) del asistente turístico, por qué se
eligió este enfoque frente a otras alternativas y cómo trabaja el modelo de
*embeddings* que sostiene la búsqueda semántica.

El código de referencia vive en `core/rag.py`, `core/config.py`,
`core/tools.py` y `core/images.py`.

---

## 1. Qué problema resuelve el RAG

El asistente debe responder dudas sobre Tenerife (playas, rutas, miradores,
gastronomía, cultura) **basándose únicamente en la guía oficial**
(`data/TENERIFE.pdf`) y no en el conocimiento paramétrico del modelo. Esto
persigue dos objetivos:

1. **Veracidad y *grounding***: el modelo solo puede afirmar lo que aparece en
   los fragmentos recuperados, lo que reduce las alucinaciones y permite
   **citar la fuente** (página y fragmento) de cada respuesta.
2. **Actualidad y mantenibilidad**: la información vive en el PDF, no en los
   pesos del modelo. Cambiar la guía no requiere reentrenar nada: basta con
   reconstruir el índice.

El prompt de sistema (`core/assistant.py`) refuerza esta política: obliga a
llamar a la herramienta de búsqueda antes de responder y prohíbe explícitamente
añadir lugares o datos que no provengan de las herramientas.

---

## 2. El pipeline RAG paso a paso

El flujo completo, desde el PDF hasta la respuesta citada, tiene seis etapas.

### 2.1. Carga del documento (*load*)

Se lee el PDF con `PyPDFLoader` (LangChain), que devuelve un `Document` por
página conservando el metadato `page`. Ese número de página es la base de las
citas y del emparejamiento posterior de fotos.

```python
docs = PyPDFLoader(str(pdf_path)).load()
```

### 2.2. Troceado en fragmentos (*split / chunking*)

El texto se divide con `RecursiveCharacterTextSplitter`:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # settings.chunk_size
    chunk_overlap=100,     # settings.chunk_overlap
    add_start_index=True,
)
chunks = splitter.split_documents(docs)
```

- **`chunk_size=500`**: fragmentos pequeños para que cada uno trate un tema
  concreto (una playa, un mirador) y el *embedding* sea semánticamente nítido.
  Trozos demasiado grandes mezclan temas y "difuminan" el vector; demasiado
  pequeños pierden contexto.
- **`chunk_overlap=100`**: un solapamiento del 20 % evita cortar frases o ideas
  justo en el límite de un fragmento, de modo que una respuesta que cae a
  caballo entre dos *chunks* siga siendo recuperable.
- **"Recursive"**: el *splitter* intenta cortar primero por separadores
  naturales (párrafos, líneas, frases) antes de partir por caracteres, lo que
  produce fragmentos más coherentes.

A cada fragmento se le añaden metadatos de cita para poder atribuirlo a su
origen:

```python
chunk.metadata["source_name"] = pdf_path.name
chunk.metadata["chunk_id"] = i
```

### 2.3. Embeddings (vectorización)

Cada fragmento se convierte en un vector numérico denso con el modelo de
*embeddings* de Gemini (`models/gemini-embedding-001`). Fragmentos con
significado parecido quedan cerca en el espacio vectorial. Esta es la pieza que
permite la **búsqueda semántica** (por significado), no por palabras exactas.
Se detalla en la sección 4.

### 2.4. Indexado en FAISS (*store*)

Los vectores se almacenan en un índice **FAISS** que se persiste en disco
(`storage/faiss_index/`):

```python
self.vector_store = FAISS.from_documents(chunks, self.embeddings)
self.vector_store.save_local(str(index_dir))
```

El índice se construye **una sola vez en el primer arranque** y luego se carga
desde disco, evitando reembeber el PDF en cada ejecución:

```python
if index_dir.exists() and not force:
    self.vector_store = FAISS.load_local(
        str(index_dir), self.embeddings,
        allow_dangerous_deserialization=True,
    )
```

> Nota: `storage/` está en `.gitignore`; el índice y las fotos se reconstruyen
> en local y **no se versionan**.

### 2.5. Recuperación (*retrieve*)

Ante una consulta, se embebe la pregunta con el **mismo modelo** y se buscan los
`top_k=5` fragmentos más cercanos por similitud:

```python
def search(self, query, k=None):
    k = k or self.settings.top_k          # 5
    return self.vector_store.similarity_search(query, k)
```

`retrieve` devuelve además las **fuentes citadas** (nombre, página, fragmento y
el *snippet* completo) y las **fotos** de las páginas recuperadas
(`GuideImageStore.images_for_pages`), de modo que la respuesta puede mostrar las
imágenes del lugar e incluir un panel de fuentes auditable.

### 2.6. Generación con citas

El contexto recuperado se formatea con encabezados de cita por fragmento:

```text
[Fuente 1: TENERIFE.pdf, página 12, fragmento 34]
<texto del fragmento>
```

Ese bloque se entrega al LLM, que redacta la respuesta **solo** a partir de él.

---

## 3. Por qué este enfoque (y no otros)

### 3.1. RAG frente a meter todo el PDF en el prompt

Se podría pegar la guía entera en el contexto del modelo. Se descartó porque:

- **Coste y latencia**: pagar y procesar todo el documento en cada turno es
  caro y lento.
- **"Lost in the middle"**: los modelos atienden peor a la información enterrada
  en contextos muy largos; recuperar solo lo relevante mejora la precisión.
- **Citas**: recuperar fragmentos concretos permite atribuir cada afirmación a
  una página; con el PDF entero esa trazabilidad se pierde.

### 3.2. RAG frente a *fine-tuning*

Entrenar (o afinar) un modelo con el contenido de la guía se descartó porque:

- **No aporta *grounding* verificable**: el conocimiento queda diluido en los
  pesos, sin posibilidad de citar la fuente.
- **Coste de actualización**: cualquier cambio en la guía exigiría reentrenar.
  Con RAG basta con reconstruir el índice (`build_index(force=True)`).
- **Riesgo de alucinación**: el *fine-tuning* enseña estilo, no garantiza
  veracidad factual.

### 3.3. RAG como **herramienta** y no cableado en el flujo

La búsqueda no está fija en el pipeline: se expone al LLM como una *tool*
(`search_tourist_guide`) que el modelo decide cuándo invocar (`core/tools.py`).
Ventajas:

- El modelo solo recupera cuando la pregunta lo necesita y puede **reformular la
  consulta** de búsqueda.
- Las citas viajan en el `artifact` de cada llamada
  (`response_format="content_and_artifact"`), de forma que cada turno refresca
  sus fuentes sin depender de estado compartido.

### 3.4. Por qué FAISS

- **Local, gratuito y rápido**: no requiere un servicio externo ni base de datos
  vectorial gestionada; encaja en un proyecto académico autocontenido.
- **Persistencia sencilla** en disco (`save_local`/`load_local`).
- **Suficiente a esta escala**: para una sola guía, una búsqueda exacta por
  similitud es instantánea; no hacen falta índices aproximados distribuidos.

### 3.5. Por qué *chunking* recursivo con solape

Equilibra **precisión** (fragmentos pequeños y temáticos) y **continuidad**
(solape que no parte ideas), como se explicó en la sección 2.2.

---

## 4. Cómo funciona el modelo de *embeddings*

### 4.1. Qué es un *embedding*

Un *embedding* es la traducción de un texto a un **vector de números reales** en
un espacio de muchas dimensiones. La propiedad clave es que la **distancia
geométrica** entre vectores refleja la **similitud semántica** entre los textos:
"playa tranquila para nadar" y "cala apacible para bañarse" caen cerca aunque no
compartan palabras.

El modelo que produce esos vectores es una red neuronal (un *transformer*)
entrenada con grandes cantidades de texto para que pares de frases con
significado parecido produzcan vectores próximos y pares con significado
distinto produzcan vectores alejados.

### 4.2. El modelo concreto: `gemini-embedding-001`

El asistente usa `models/gemini-embedding-001` a través de
`GoogleGenerativeAIEmbeddings` (`langchain-google-genai`). Se configura en
`core/config.py` (`embedding_model`) y puede cambiarse con la variable de
entorno `EMBEDDING_MODEL`. La clave de API (`GOOGLE_API_KEY`) la detecta
automáticamente la librería.

```python
self.embeddings = GoogleGenerativeAIEmbeddings(model=settings.embedding_model)
```

### 4.3. Dos usos del mismo modelo

El **mismo** modelo de *embeddings* se usa en dos momentos, y eso es esencial:

1. **Indexado**: embebe cada fragmento del PDF (vectores guardados en FAISS).
2. **Consulta**: embebe la pregunta del usuario en el momento de buscar.

Solo así pregunta y fragmentos viven en el **mismo espacio vectorial** y la
comparación de distancias tiene sentido. Si se cambia el modelo de *embeddings*,
hay que **reconstruir el índice** (`force=True`), porque los vectores antiguos
no son comparables con los nuevos.

### 4.4. La búsqueda por similitud

`similarity_search` embebe la consulta y recupera los `k` vectores más cercanos
del índice. FAISS calcula la cercanía con una métrica de distancia (por defecto,
distancia euclídea L2; conceptualmente equivalente a buscar los vectores que más
"apuntan" en la misma dirección que la consulta). Los `k=5` fragmentos
resultantes son los que se entregan al LLM como contexto.

### 4.5. Implicaciones prácticas

- **Idioma**: el modelo es multilingüe, así que una pregunta en español recupera
  bien fragmentos en español de la guía.
- **Calidad de los *chunks***: como la unidad de recuperación es el fragmento,
  el *chunking* (sección 2.2) condiciona directamente la calidad del *embedding*
  y, por tanto, de la búsqueda.
- **Reconstrucción**: cambiar de modelo de *embeddings* o de parámetros de
  troceado obliga a regenerar el índice.

### 4.6. Embeddings locales frente a embeddings vía API (lo que se vio en los notebooks)

El proyecto genera los *embeddings* llamando a un modelo alojado de Google
(`gemini-embedding-001`). No es la única forma: en los notebooks del máster
(`resources/Large Lenguage Models/`) aparecen **otros mecanismos para producir
embeddings de forma local**, sin depender de la API de Gemini. Conviene
contrastarlos para entender por qué aquí se eligió la vía API.

**Lo que muestran los notebooks:**

- **Sesión 01 (`sesion_01/Transformer.ipynb`)**: construye un *Transformer*
  desde cero y genera los *embeddings* **en local**, con una capa entrenable
  `tf.keras.layers.Embedding(vocab_size, d_model)` sumada a una **codificación
  posicional** (`positional_encoding`). Aquí los vectores no vienen de ningún
  servicio externo: se aprenden como parte del modelo durante el entrenamiento.
- **Sesión 03 (`sesion_03/sesion_03.ipynb`)**: el notebook de RAG ya usa
  `GoogleGenerativeAIEmbeddings` (Gemini) con `InMemoryVectorStore`, es decir, el
  **mismo enfoque por API** que adopta este proyecto.

**Matiz clave — no son lo mismo.** Los *embeddings* locales de la sesión 01 son
**embeddings de token** aprendidos para una tarea concreta (traducción): su
trabajo es alimentar las capas de atención del propio Transformer, no medir la
similitud semántica entre dos textos cualesquiera. Los *embeddings* de
`gemini-embedding-001` son **embeddings de frase/documento**: un modelo ya
preentrenado y optimizado para que la **distancia** entre vectores refleje la
**cercanía de significado**, que es justo lo que necesita la búsqueda del RAG.
Dicho de otro modo: la sesión 01 enseña *cómo nace un embedding por dentro*; el
RAG necesita un embedding *ya entrenado para buscar*.

**Alternativas locales reales para un RAG.** Si se quisiera evitar la API, la
opción equivalente no sería la capa de la sesión 01, sino un modelo de
*embeddings de frase* preentrenado que corra en la propia máquina —por ejemplo
`sentence-transformers`/`HuggingFaceEmbeddings` (modelos tipo `all-MiniLM`)— o,
en su versión más simple y clásica, una representación dispersa tipo **TF-IDF**
(coincidencia léxica, sin semántica).

**Por qué este proyecto usa la API y no embeddings locales:**

- **Coherencia de proveedor**: la generación ya es Gemini; usar embeddings de
  Gemini mantiene un solo proveedor y una sola clave (criterio explicitado en el
  notebook del proyecto).
- **Calidad sin entrenar nada**: el modelo está preentrenado y es multilingüe;
  funciona bien en español desde el primer momento, sin recolectar datos ni
  entrenar una capa propia (lo que sí exigiría el enfoque de la sesión 01).
- **Simplicidad operativa**: no hay que descargar pesos, gestionar GPU/CPU ni
  fijar versiones de un modelo local; basta una llamada de red.
- **Escala adecuada**: para una sola guía, el coste de embeber por API es
  mínimo y ocurre **una sola vez** al construir el índice.

**Cuándo tendría sentido lo local**: privacidad estricta (que el texto no salga
de la máquina), funcionamiento sin conexión, evitar costes/cuotas de API o
necesitar control total sobre el modelo. En esos casos, cambiar a un *embedder*
local es directo —basta sustituir `GoogleGenerativeAIEmbeddings` por el
*embedder* elegido en `TouristGuideRAG.__init__`— **reconstruyendo el índice**
(`force=True`), ya que vectores de modelos distintos no son comparables
(sección 4.3).

---

## 5. Resumen del flujo

```text
TENERIFE.pdf
   │  PyPDFLoader (1 doc por página)
   ▼
Fragmentos (chunk_size=500, overlap=100) + metadatos de cita
   │  gemini-embedding-001
   ▼
Vectores  ──►  Índice FAISS  (storage/faiss_index/, persistente)
                   ▲
Consulta del usuario ──► mismo modelo de embeddings ──► similarity_search(k=5)
                                                              │
                                                              ▼
                          5 fragmentos + fuentes citadas + fotos por página
                                                              │
                                                              ▼
                                   LLM (Gemini) redacta la respuesta SOLO con
                                   ese contexto, con grounding y citas
```

---

## 6. Parámetros clave (en `core/config.py`)

| Parámetro          | Valor por defecto             | Efecto                                            |
| ------------------ | ----------------------------- | ------------------------------------------------- |
| `embedding_model`  | `models/gemini-embedding-001` | Modelo que vectoriza fragmentos y consultas       |
| `chunk_size`       | `500`                         | Tamaño (caracteres) de cada fragmento             |
| `chunk_overlap`    | `100`                         | Solape entre fragmentos contiguos                 |
| `top_k`            | `5`                           | Nº de fragmentos recuperados por consulta         |
| `index_dir`        | `storage/faiss_index/`        | Ruta del índice FAISS persistente                 |

Todos pueden ajustarse; recuerda que cambiar `embedding_model`, `chunk_size` o
`chunk_overlap` requiere reconstruir el índice con `build_index(force=True)`.
