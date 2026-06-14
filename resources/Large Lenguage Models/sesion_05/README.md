# Entorno para `sesion_05.ipynb`

Este directorio contiene todo lo necesario para ejecutar la sesión de agentes con LLMs en local.

El notebook principal es:

```text
sesion_05.ipynb
```

## API key

El notebook usa la API de OpenAI.

Crea un archivo `.env` a partir de la plantilla:

```bash
cp .env.template .env
```

Rellena:

```text
OPENAI_API_KEY="..."
```

Puedes cambiar los modelos con:

```text
OPENAI_GENERATION_MODEL="gpt-4.1-mini"
OPENAI_GUARDRAIL_MODEL="gpt-4.1-mini"
```

## macOS y Linux

Requisitos:

- `make`.
- Conexión a internet en la primera instalación.

No hace falta instalar Python manualmente: `uv` instalará Python `3.11.9` y creará el entorno virtual.

Configura todo con:

```bash
make setup
```

Abre el notebook con:

```bash
make notebook
```

En Jupyter, selecciona el kernel:

```text
Python (pontia-sesion-05)
```

## Windows

En Windows nativo puedes usar Python `3.11` y `requirements.txt`.

Desde esta carpeta:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name pontia-sesion-05 --display-name "Python (pontia-sesion-05)"
jupyter lab sesion_05.ipynb
```

## Instalación manual en macOS/Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name pontia-sesion-05 --display-name "Python (pontia-sesion-05)"
jupyter lab sesion_05.ipynb
```

## Datos incluidos

La carpeta contiene datos locales para los ejemplos:

- `data.csv`: métricas comerciales tabulares.
- `data/company_docs_src/`: documentación interna mínima para herramientas de búsqueda.
- `data/eval_cases.jsonl`: casos de evaluación.
- `data/finetune_style_train.jsonl` y `data/finetune_style_validation.jsonl`: ejemplos de estilo.

## Dependencias principales

- `openai`
- `openai-agents`
- `python-dotenv`
- `pydantic`
- `pandas`
- `graphviz`
- `jupyterlab`
- `ipykernel`

## Comandos útiles

```bash
make check
```

Verifica imports y versiones principales.

```bash
make lock
```

Actualiza `uv.lock` tras cambiar dependencias.

```bash
make clean
```

Elimina `.venv`.
