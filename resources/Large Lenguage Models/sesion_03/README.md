# Entorno para `sesion_03.ipynb`

Este directorio contiene la configuración mínima para ejecutar la sesión 03 sobre RAG con LLMs.

El notebook principal recomendado es:

```text
sesion_03.ipynb
```

## API Key

El notebook usa la API de Google Gemini.

Antes de ejecutarlo en local, crea un archivo `.env` a partir de `.env.template`:

```bash
cp .env.template .env
```

Rellena:

```text
GOOGLE_API_KEY="..."
```

En Windows PowerShell, si no quieres copiar el fichero manualmente:

```powershell
Copy-Item .env.template .env
```

## macOS y Linux

Esta es la ruta recomendada para macOS y Linux.

Requisitos:

- `make`.
- Conexión a internet en la primera instalación.

No hace falta instalar Python manualmente: `uv` instalará Python `3.11.9` y creará el entorno virtual.

### Instalar `make`

Comprueba si ya lo tienes:

```bash
make --version
```

Si no está instalado:

- macOS:
  ```bash
  xcode-select --install
  ```

- Ubuntu/Debian:
  ```bash
  sudo apt update
  sudo apt install make
  ```

- Fedora:
  ```bash
  sudo dnf install make
  ```

- Arch Linux:
  ```bash
  sudo pacman -S make
  ```

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
Python (pontia-sesion-03)
```

## Windows

En Windows nativo puedes usar Python `3.11` y `requirements.txt`.

Comprueba si tienes Python 3.11:

```powershell
py -3.11 --version
```

Si no lo tienes, instala Python 3.11 de 64 bits desde:

```text
https://www.python.org/downloads/
```

Durante la instalación, marca **Add python.exe to PATH**. Después cierra y vuelve a abrir PowerShell.

Desde la carpeta del proyecto:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name pontia-sesion-03 --display-name "Python (pontia-sesion-03)"
jupyter lab sesion_03.ipynb
```

En Jupyter, selecciona el kernel:

```text
Python (pontia-sesion-03)
```

## Instalación manual en macOS/Linux

Si no quieres usar `make`, también puedes instalar con `venv` y `pip`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name pontia-sesion-03 --display-name "Python (pontia-sesion-03)"
jupyter lab sesion_03.ipynb
```

## Dependencias principales

El notebook usa:

- `google-genai`
- `langchain`
- `langchain-community`
- `langchain-core`
- `langchain-google-genai`
- `langchain-text-splitters`
- `langgraph`
- `pypdf`
- `python-dotenv`
- `jupyterlab`
- `ipykernel`

`uv.lock` fija las dependencias transitivas para la ruta macOS/Linux con `make setup`. `requirements.txt` contiene una exportación fijada de dependencias para instalación manual con `pip`, especialmente útil en Windows.

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
