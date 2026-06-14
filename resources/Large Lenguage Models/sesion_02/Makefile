SHELL := /bin/bash

PYTHON_VERSION := 3.11.9
UV_VERSION := 0.7.13
VENV := .venv
PYTHON := $(VENV)/bin/python
KERNEL_NAME := pontia-sesion-02
export XDG_CACHE_HOME := $(CURDIR)/.cache
export MPLCONFIGDIR := $(CURDIR)/.matplotlib-cache

.PHONY: help setup ensure-uv python lock sync kernel check notebook clean

help:
	@echo "Comandos disponibles:"
	@echo "  make setup    Crea el entorno completo y registra el kernel de Jupyter"
	@echo "  make check    Verifica imports y versiones principales"
	@echo "  make notebook Abre JupyterLab con el entorno del proyecto"
	@echo "  make lock     Actualiza uv.lock"
	@echo "  make clean    Elimina .venv"

setup: ensure-uv python sync kernel check
	@echo "Entorno listo. Usa el kernel '$(KERNEL_NAME)' en sesion_02.ipynb."

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
	@$(PYTHON) -c 'import importlib, platform; from importlib.metadata import version; importlib.import_module("openai"); importlib.import_module("google.genai"); importlib.import_module("langchain"); importlib.import_module("langchain_core"); importlib.import_module("langgraph"); importlib.import_module("tiktoken"); importlib.import_module("dotenv"); packages = {"Python": platform.python_version(), "openai": version("openai"), "google-genai": version("google-genai"), "langchain": version("langchain"), "langchain-core": version("langchain-core"), "langchain-google-genai": version("langchain-google-genai"), "langchain-openai": version("langchain-openai"), "langgraph": version("langgraph"), "python-dotenv": version("python-dotenv"), "tiktoken": version("tiktoken"), "ipykernel": version("ipykernel")}; [print(f"{name}: {pkg_version}") for name, pkg_version in packages.items()]'

notebook:
	@uv run --frozen --python $(PYTHON_VERSION) jupyter lab sesion_02.ipynb

clean:
	@rm -rf "$(VENV)"
