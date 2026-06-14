SHELL := /bin/bash

PYTHON_VERSION := 3.11.9
UV_VERSION := 0.7.13
VENV := .venv
PYTHON := $(VENV)/bin/python
KERNEL_NAME := pontia-transformer
export XDG_CACHE_HOME := $(CURDIR)/.cache
export MPLCONFIGDIR := $(CURDIR)/.matplotlib-cache

.PHONY: help setup check-platform ensure-uv python lock sync kernel check notebook clean

help:
	@echo "Comandos disponibles:"
	@echo "  make setup    Crea el entorno completo y registra el kernel de Jupyter"
	@echo "  make check    Verifica imports y versiones principales"
	@echo "  make notebook Abre JupyterLab con el entorno del proyecto"
	@echo "  make lock     Actualiza uv.lock"
	@echo "  make clean    Elimina .venv"

setup: check-platform ensure-uv python sync kernel check
	@echo "Entorno listo. Usa el kernel '$(KERNEL_NAME)' en Transformer.ipynb."

check-platform:
	@if [ "$$(uname -m)" != "x86_64" ]; then \
		echo "Arquitectura no soportada por este entorno fijado: $$(uname -m)"; \
		echo "tensorflow-text==2.16.1 solo ofrece wheels compatibles para x86_64 en este setup."; \
		exit 1; \
	fi

ensure-uv:
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "uv no está instalado. Instalando uv==$(UV_VERSION) con pip..."; \
		python3 -m pip install --user "uv==$(UV_VERSION)"; \
	fi
	@uv --version

python: ensure-uv
	@uv python install $(PYTHON_VERSION)

lock: ensure-uv
	@uv lock --python $(PYTHON_VERSION)

sync: ensure-uv
	@mkdir -p "$(MPLCONFIGDIR)" "$(XDG_CACHE_HOME)"
	@uv sync --frozen --python $(PYTHON_VERSION)

kernel:
	@$(PYTHON) -m ipykernel install --user --name "$(KERNEL_NAME)" --display-name "Python ($(KERNEL_NAME))"

check:
	@mkdir -p "$(MPLCONFIGDIR)" "$(XDG_CACHE_HOME)"
	@$(PYTHON) -c 'import importlib, platform; packages = {"Python": platform.python_version(), "tensorflow": importlib.import_module("tensorflow").__version__, "tensorflow_text": importlib.import_module("tensorflow_text").__version__, "tensorflow_datasets": importlib.import_module("tensorflow_datasets").__version__, "numpy": importlib.import_module("numpy").__version__, "matplotlib": importlib.import_module("matplotlib").__version__, "ipykernel": importlib.import_module("ipykernel").__version__}; [print(f"{name}: {version}") for name, version in packages.items()]'

notebook:
	@uv run --frozen --python $(PYTHON_VERSION) jupyter lab Transformer.ipynb

clean:
	@rm -rf "$(VENV)"
