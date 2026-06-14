# Entorno para `sesion_02.ipynb`

Este directorio contiene la configuración mínima para ejecutar `sesion_02.ipynb`.

## Opción rápida: Google Colab

Todo lo siguiente aplica si quieres ejecutar el notebook en local.

Si no quieres dedicar tiempo a configurar Python, entornos virtuales y dependencias, puedes abrir `sesion_02.ipynb` directamente en Google Colab y ejecutarlo allí.

## API Keys

El notebook usa APIs de OpenAI y Google Gemini.

Antes de ejecutarlo en local, crea un archivo `.env` a partir de `.env.template`:

```bash
cp .env.template .env
```

Rellena:

```text
OPENAI_API_KEY="..."
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
Python (pontia-sesion-02)
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
python -m ipykernel install --user --name pontia-sesion-02 --display-name "Python (pontia-sesion-02)"
jupyter lab sesion_02.ipynb
```

En Jupyter, selecciona el kernel:

```text
Python (pontia-sesion-02)
```

## Instalación manual en macOS/Linux

Si no quieres usar `make`, también puedes instalar con `venv` y `pip`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name pontia-sesion-02 --display-name "Python (pontia-sesion-02)"
jupyter lab sesion_02.ipynb
```

## Dependencias principales

El notebook usa:

- `openai`
- `google-genai`
- `langchain`
- `langchain-openai`
- `langchain-google-genai`
- `langgraph`
- `python-dotenv`
- `tiktoken`
- `jupyterlab`
- `ipykernel`

`uv.lock` fija las dependencias transitivas para la ruta macOS/Linux con `make setup`. `requirements.txt` contiene una exportación reproducible para instalación manual con `pip`.

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
