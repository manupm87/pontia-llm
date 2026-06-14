# Entorno para `Transformer.ipynb`

Este directorio contiene dos formas de configurar el entorno del notebook:

- **macOS/Linux:** entorno reproducible con `pyproject.toml`, `uv.lock` y `Makefile`.
- **Windows:** entorno manual con `venv` y `requirements-windows.txt`, usando versiones compatibles con Windows nativo.

## Opción rápida: Google Colab

Todo lo que sigue aplica si quieres ejecutar el notebook en local.

La configuración local tiene cierta complejidad porque TensorFlow y TensorFlow Text publican wheels distintos según sistema operativo, arquitectura y versión de Python. Por eso hay dependencias diferentes para macOS/Linux y Windows.

Si no quieres dedicar tiempo a configurar el entorno local, la opción más sencilla es abrir `Transformer.ipynb` directamente en Google Colab y ejecutarlo allí. Colab ya proporciona un entorno preparado para notebooks y evita la mayoría de problemas de instalación.

## macOS y Linux

Esta es la ruta recomendada para macOS y Linux.

Requisitos:

- `make`.
- Conexión a internet en la primera instalación.
- Arquitectura `x86_64/amd64`.

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
  Esto instala las Command Line Tools de Apple, que incluyen `make`.

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

El comando:

- instala `uv==0.7.13` si no existe;
- instala Python `3.11.9`;
- crea `.venv`;
- instala las dependencias exactas fijadas en `uv.lock`;
- registra el kernel `Python (pontia-transformer)`;
- valida los imports principales.

Abre el notebook con:

```bash
make notebook
```

En Jupyter, selecciona el kernel:

```text
Python (pontia-transformer)
```

## Windows

En Windows nativo usa Python `3.10` y el fichero `requirements-windows.txt`.

Requisitos:

- Python `3.10.x` de 64 bits.
- PowerShell o terminal equivalente.

Comprueba si tienes Python 3.10 disponible:

```powershell
py -3.10 --version
```

Si ese comando falla, instala Python 3.10 desde la web oficial:

```text
https://www.python.org/downloads/release/python-31011/
```

Durante la instalación, marca la opción **Add python.exe to PATH**. Después cierra y vuelve a abrir PowerShell y comprueba de nuevo:

```powershell
py -3.10 --version
```

Desde la carpeta del proyecto:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-windows.txt
python -m ipykernel install --user --name pontia-transformer --display-name "Python (pontia-transformer)"
jupyter lab Transformer.ipynb
```

En Jupyter, selecciona el kernel:

```text
Python (pontia-transformer)
```

La ruta Windows usa TensorFlow `2.10.1` y TensorFlow Text `2.10.0` porque son versiones con soporte de wheels para Windows. Las versiones más recientes usadas en macOS/Linux no están disponibles de forma equivalente para Windows nativo.

## Instalación Manual en macOS/Linux

Si no quieres usar `make`, también puedes crear el entorno manualmente en plataformas compatibles:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name pontia-transformer --display-name "Python (pontia-transformer)"
jupyter lab Transformer.ipynb
```

## Dependencias

Para macOS/Linux, las dependencias directas están fijadas en `pyproject.toml` y `requirements.txt`:

- Python `3.11`
- TensorFlow `2.16.2`
- TensorFlow Text `2.16.1`
- TensorFlow Datasets `4.9.6`
- NumPy `1.26.4`
- Matplotlib `3.8.4`
- Protobuf `4.25.3`
- Importlib Resources `6.4.5`
- JupyterLab `4.2.5`
- IPython Kernel `6.29.5`

Para Windows, las dependencias están fijadas en `requirements-windows.txt`:

- Python `3.10`
- TensorFlow `2.10.1`
- TensorFlow Text `2.10.0`
- TensorFlow Datasets `4.9.0`
- NumPy `1.26.4`
- Matplotlib `3.8.4`
- Protobuf `3.19.6`
- JupyterLab `4.2.5`
- IPython Kernel `6.29.5`

`uv.lock` fija además las dependencias transitivas de la ruta macOS/Linux para que `make setup` sea reproducible.

## Comandos Útiles

```bash
make check
```

Verifica imports y versiones principales en la ruta macOS/Linux.

```bash
make lock
```

Actualiza `uv.lock` tras cambiar dependencias.

```bash
make clean
```

Elimina `.venv`.
