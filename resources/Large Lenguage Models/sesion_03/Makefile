SHELL := /bin/bash

PYTHON_VERSION := 3.11.9
UV_VERSION := 0.7.13
VENV := .venv
PYTHON := $(VENV)/bin/python
KERNEL_NAME := pontia-sesion-03
NOTEBOOK := sesion_03.ipynb
UV := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
export XDG_CACHE_HOME := $(CURDIR)/.cache
export MPLCONFIGDIR := $(CURDIR)/.matplotlib-cache
export UV_CACHE_DIR := $(CURDIR)/.cache/uv
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.uv-python

.PHONY: help setup ensure-uv python lock sync kernel check notebook clean

help:
	@echo "Comandos disponibles:"
	@echo "  make setup    Crea el entorno completo y registra el kernel de Jupyter"
	@echo "  make check    Verifica imports y versiones principales"
	@echo "  make notebook Abre JupyterLab con el entorno del proyecto"
	@echo "  make lock     Actualiza uv.lock"
	@echo "  make clean    Elimina .venv"

setup: ensure-uv python sync kernel check
	@echo "Entorno listo. Usa el kernel '$(KERNEL_NAME)' en $(NOTEBOOK)."

ensure-uv:
	@if ! command -v uv >/dev/null 2>&1 && [ ! -x "$(HOME)/.local/bin/uv" ]; then \
		echo "uv no está instalado. Instalando uv==$(UV_VERSION) con pip..."; \
		python3 -m pip install --user "uv==$(UV_VERSION)"; \
	fi
	@$(UV) --version

python: ensure-uv
	@$(UV) python install $(PYTHON_VERSION)

lock: ensure-uv
	@$(UV) lock --python $(PYTHON_VERSION)

sync: ensure-uv
	@mkdir -p "$(MPLCONFIGDIR)" "$(XDG_CACHE_HOME)"
	@$(UV) sync --frozen --python $(PYTHON_VERSION)

kernel:
	@$(PYTHON) -m ipykernel install --user --name "$(KERNEL_NAME)" --display-name "Python ($(KERNEL_NAME))"

check:
	@mkdir -p "$(MPLCONFIGDIR)" "$(XDG_CACHE_HOME)"
	@$(PYTHON) -c 'import importlib, platform; from importlib.metadata import version; importlib.import_module("google.genai"); importlib.import_module("langchain"); importlib.import_module("langchain_community"); importlib.import_module("langchain_core"); importlib.import_module("langchain_google_genai"); importlib.import_module("langchain_text_splitters"); importlib.import_module("langgraph"); importlib.import_module("pypdf"); importlib.import_module("dotenv"); packages = {"Python": platform.python_version(), "google-genai": version("google-genai"), "langchain": version("langchain"), "langchain-community": version("langchain-community"), "langchain-core": version("langchain-core"), "langchain-google-genai": version("langchain-google-genai"), "langchain-text-splitters": version("langchain-text-splitters"), "langgraph": version("langgraph"), "pypdf": version("pypdf"), "python-dotenv": version("python-dotenv"), "ipykernel": version("ipykernel")}; [print(f"{name}: {pkg_version}") for name, pkg_version in packages.items()]'

notebook:
	@$(UV) run --frozen --python $(PYTHON_VERSION) jupyter lab $(NOTEBOOK)

clean:
	@rm -rf "$(VENV)"
