# Asistente turístico de Tenerife

Asistente conversacional que ayuda a los turistas a planificar su viaje a
Tenerife. Responde en español, de forma cercana y práctica, combinando **RAG**
sobre la guía oficial, **function calling** (tiempo, mar y fechas) y **diálogo
multiturno** con memoria. Es el trabajo de fin de módulo LLM (Pontia).

**Qué sabe hacer:**

- **RAG sobre la guía oficial** (`data/TENERIFE.pdf`): recupera información sobre
  playas, rutas, gastronomía, cultura y lugares de interés, **basándose solo en
  los fragmentos recuperados** y citando la fuente. La guía se consulta en cada
  turno, también en las preguntas de seguimiento.
- **Fotos de los lugares**: extrae las fotografías del PDF y las **intercala** en
  la respuesta junto al lugar que mencionan.
- **Tiempo y mar** (`get_weather`, `get_sea_conditions`): previsión y estado del
  mar (oleaje) para una fecha, vía las APIs públicas de Open-Meteo, con respaldo
  simulado ante fallos de red.
- **Fechas inteligentes** (`resolve_date`): traduce "hoy", "mañana", "este finde"
  o "el miércoles" a una fecha exacta para las herramientas de tiempo y mar.
- **Diálogo multiturno** con memoria y recorte automático del historial.
- **Razonamiento en streaming**: muestra el "thinking" de Gemini y la respuesta en
  vivo.
- **Guardarraíles**: detección de manipulación de instrucciones (siempre) y
  filtros LLM opcionales de tema y fidelidad.
- **Interfaz cuidada**: tema visual propio, ejemplos clicables, actividad de las
  herramientas en vivo, fuentes citadas, galería de fotos y exportación del chat.

Construido sobre **Google Gemini** orquestado con **LangChain**, con **FAISS**
como vector store y **PyMuPDF** para extraer las imágenes de la guía.

> 🚀 **Demo en vivo:** prueba el asistente desplegado en
> **<https://pontia-llm-zrvfwbe9t6m28chhrn6cms.streamlit.app/>** (Streamlit
> Community Cloud) — necesita una `GOOGLE_API_KEY` válida configurada en la app.
>
> 📄 **¿Cómo está diseñado por dentro?** La arquitectura, las decisiones técnicas,
> la evaluación y la trazabilidad de los requisitos del enunciado están en
> **[INFORME.md](INFORME.md)**.
>
> 📓 **¿Lo quieres ver en acción paso a paso?** El **[notebook de
> demostración](notebook_asistente_tenerife.ipynb)** recorre todas las capacidades
> de principio a fin (ver [cómo ejecutarlo](#cómo-ejecutarlo)).

## Instalación

Necesitas **Python 3.11+**.

### Con Make (recomendado)

```bash
make setup        # crea .venv, instala dependencias y verifica el entorno
```

### Manual

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuración de la API key

Copia la plantilla de variables de entorno y rellena tu clave de Google (Gemini):

```bash
cp .env.template .env
```

```ini
GOOGLE_API_KEY="tu_clave_aqui"
```

La clave se carga automáticamente con `python-dotenv` y la detecta
`langchain-google-genai`.

## Cómo ejecutarlo

¿Solo quieres probarlo? Usa la **[demo en vivo](https://pontia-llm-zrvfwbe9t6m28chhrn6cms.streamlit.app/)**
sin instalar nada. Para ejecutarlo en local, cada tarea tiene un atajo de `make`
y su comando equivalente:

| Tarea                          | Con Make        | Comando directo                              |
|--------------------------------|-----------------|----------------------------------------------|
| Aplicación web (Streamlit)     | `make run`      | `streamlit run app.py`                       |
| Notebook de demostración       | `make notebook` | `jupyter lab notebook_asistente_tenerife.ipynb` |
| Tests                          | `make test`     | `python -m pytest`                           |
| Evaluación (genera CSV + PNG)  | `make eval`     | `python -m scripts.run_eval`                 |

`make help` lista todos los comandos disponibles.

- La **aplicación web** ofrece el chat, preguntas de ejemplo clicables, respuesta
  en *streaming*, la actividad de las herramientas en vivo, las fotos de la guía,
  el panel de fuentes y la exportación de la conversación. Desde la barra lateral
  se ajustan **en vivo** los parámetros del modelo, el *streaming*, el razonamiento
  y los guardarraíles avanzados.
- El **[notebook de demostración](notebook_asistente_tenerife.ipynb)** recorre cada
  capacidad en orden natural (13 secciones): indexado y recuperación con citas,
  fotos e intercalado, las funciones externas (tiempo, mar y fechas), el asistente
  multiturno, *streaming* y razonamiento, guardarraíles, evaluación
  (*LLM-as-judge*) y observabilidad. Ejecútalo con *Run All* tras configurar `.env`.
- Los **tests** no requieren red ni las dependencias pesadas (usan dobles de
  prueba para el LLM y el RAG).
- La **evaluación** requiere `GOOGLE_API_KEY` y escribe `storage/eval_results.csv`
  y `storage/eval_summary.png` (ver detalle en [INFORME.md](INFORME.md#4-evaluación)).

## Parámetros configurables por entorno

Se ajustan en `.env`; si no se definen, se usan los valores por defecto:

| Variable             | Valor por defecto              | Descripción                          |
|----------------------|--------------------------------|--------------------------------------|
| `GOOGLE_API_KEY`     | —                              | Clave de la API de Google (Gemini)   |
| `GENERATION_MODEL`   | `gemini-2.5-flash-lite`        | Modelo de generación                 |
| `EMBEDDING_MODEL`    | `models/gemini-embedding-001`  | Modelo de embeddings                 |
| `TEMPERATURE`        | `0.2`                          | Temperatura de muestreo              |
| `TOP_P`              | `0.95`                         | Núcleo de probabilidad (top-p)       |
| `MAX_OUTPUT_TOKENS`  | `1024`                         | Límite de tokens de la respuesta     |
| `THINKING_BUDGET`    | `1024`                         | Presupuesto de razonamiento (0 = off, -1 = dinámico) |

El resto de parámetros (troceado, `top_k`, longitud del historial, rutas, tamaño
mínimo de imagen y número máximo de fotos) viven en `core/config.py`
(clase `Settings`).

## Índice FAISS e imágenes

El índice vectorial y las fotos de la guía se **construyen en la primera
ejecución** a partir de `data/TENERIFE.pdf` y se persisten en `storage/`
(`faiss_index/` e `images/` con un `manifest.json`). En arranques posteriores se
cargan desde disco. Para forzar su reconstrucción (tras actualizar el PDF o
cambiar el modelo de embeddings), elimina el directorio `storage/` o invoca
`build_index(force=True)`. Todo `storage/` está *gitignored*.

## Estructura del proyecto

```text
.
├── app.py / ui_theme.py       # Interfaz web (Streamlit) y tema visual
├── core/                      # Paquete reutilizable del asistente (ver INFORME.md)
├── scripts/run_eval.py        # Ejecuta la evaluación y guarda tabla + gráfico
├── tests/                     # Suite de tests (pytest)
├── data/TENERIFE.pdf          # Guía oficial usada como fuente del RAG
├── storage/                   # Índice FAISS + fotos (generado, gitignored)
├── notebook_asistente_tenerife.ipynb   # Notebook de demostración
├── Makefile                   # Atajos: setup / run / test / eval / notebook
└── requirements.txt
```
