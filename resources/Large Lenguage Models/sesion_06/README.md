# Entorno para `sesion_06.ipynb`

Este directorio contiene todo lo necesario para ejecutar la sesión de arquitecturas multiagénticas en local.

El notebook principal es:

```text
sesion_06.ipynb
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
OPENAI_FAST_MODEL="gpt-4.1-mini"
```

## macOS y Linux

Requisitos:

- `make`.
- Graphviz de sistema para renderizar diagramas (`dot`). En macOS: `brew install graphviz`. En Ubuntu/Debian: `sudo apt-get install graphviz`.
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
Python (pontia-sesion-06)
```

## Windows

En Windows nativo puedes usar Python `3.11` y `requirements.txt`.
Instala también Graphviz y asegúrate de que `dot` esté disponible en el `PATH`.

Desde esta carpeta:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name pontia-sesion-06 --display-name "Python (pontia-sesion-06)"
jupyter lab sesion_06.ipynb
```

## Instalación manual en macOS/Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name pontia-sesion-06 --display-name "Python (pontia-sesion-06)"
jupyter lab sesion_06.ipynb
```

## Contenido de la sesión

El notebook trabaja arquitecturas multiagénticas sobre un caso autocontenido de operaciones internas:

- Router con handoffs hacia especialistas.
- Supervisor con subagentes usados como herramientas.
- Arquitectura jerárquica con agentes que coordinan otros subagentes.
- Patrón planner/executor con revisión humana antes de ejecutar.
- Servidores MCP remotos, conectores MCP y servidor MCP local con `stdio`.
- Skills cargadas desde carpetas reales con `SKILL.md`, `references/`, `scripts/` y `assets/`.
- Comparativa de uso real de tokens y observabilidad básica de las ejecuciones.

Los datos de ejemplo viven en `operations_support.py`. El servidor MCP local está en `mcp_servers/operations_mcp_server.py` y las skills de ejemplo están en `skills/operaciones/`.

## Dependencias principales

- `openai`
- `openai-agents`
- `mcp`
- `python-dotenv`
- `pydantic`
- `pandas`
- `graphviz`
- `tiktoken`
- `jupyterlab`
- `ipykernel`

## Comandos útiles

```bash
make check
```

Verifica imports, versiones principales, binario `dot`, render básico de Graphviz, archivos auxiliares y validez del notebook.

```bash
make lock
```

Actualiza `uv.lock` tras cambiar dependencias.

```bash
make clean
```

Elimina `.venv` y cachés locales generadas durante la ejecución.
