# Ops Copilot

Aplicación Streamlit que simula un asistente interno de operaciones con herramientas auditables. El copilot puede consultar documentación interna, inventario, métricas comerciales y crear borradores de gasto pendientes de confirmación humana.

## Funcionalidad

- Chat operativo con selección automática u obligatoria de herramientas.
- Ejecutor con validación local de argumentos mediante JSON Schema.
- Trazas por llamada: herramienta, argumentos, estado, salida y latencia.
- RAG como herramienta sobre una base documental local.
- Análisis de datos tabulares desde `data/business_metrics.csv`.
- Inventario local desde `data/inventory.json`.
- Borradores de gasto con confirmación o rechazo desde la interfaz.

## Requisitos

- Python 3.11 o superior.
- Una clave de OpenAI en `OPENAI_API_KEY`.

## Instalación

```bash
cd sesion_04/ops_copilot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.template .env
```

Edita `.env` y rellena `OPENAI_API_KEY`.

## Ejecución

```bash
streamlit run app.py
```

También puedes usar:

```bash
make setup
make run
```

## Configuración

Variables disponibles:

```text
OPENAI_API_KEY
OPENAI_GENERATION_MODEL
OPENAI_EMBEDDING_MODEL
OPENAI_TIMEOUT_SECONDS
OPS_DEFAULT_TOP_K
```

Los datos de demostración están incluidos en la carpeta `data/`, por lo que la aplicación no depende de rutas externas.
