SHELL := /bin/bash

PYTHON_VERSION := 3.11.9
UV_VERSION := 0.7.13
VENV := .venv
PYTHON := $(VENV)/bin/python
KERNEL_NAME := pontia-sesion-06
NOTEBOOK := sesion_06.ipynb
UV := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
export XDG_CACHE_HOME := $(CURDIR)/.cache
export MPLCONFIGDIR := $(CURDIR)/.matplotlib-cache
export UV_CACHE_DIR := $(CURDIR)/.cache/uv
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.uv-python

.PHONY: help setup ensure-uv python lock sync kernel check notebook clean

help:
	@echo "Comandos disponibles:"
	@echo "  make setup    Crea el entorno completo y registra el kernel de Jupyter"
	@echo "  make check    Verifica entorno, imports, Graphviz y notebook"
	@echo "  make notebook Abre JupyterLab con el entorno del proyecto"
	@echo "  make lock     Actualiza uv.lock"
	@echo "  make clean    Elimina .venv y cachés locales"

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
	@command -v dot >/dev/null || (echo "Falta Graphviz de sistema: instala el binario 'dot' antes de renderizar diagramas." >&2; exit 1)
	@$(PYTHON) -c 'import ast, importlib, nbformat, platform; from graphviz import Source; from importlib.metadata import version; from pathlib import Path; modules = ["agents", "agents.mcp", "agents.run", "agents.sandbox", "agents.sandbox.capabilities", "agents.sandbox.entries", "agents.sandbox.sandboxes.unix_local", "dotenv", "graphviz", "mcp", "mcp.server.fastmcp", "openai", "operations_support", "pandas", "pydantic", "tiktoken"]; [importlib.import_module(module) for module in modules]; [(_ for _ in ()).throw(SystemExit(f"Falta {path}")) for path in [Path("sesion_06.ipynb"), Path("operations_support.py"), Path("mcp_servers/operations_mcp_server.py"), Path("skills/operaciones")] if not path.exists()]; Source("digraph G { A -> B }").pipe(format="svg"); notebook = nbformat.read("sesion_06.ipynb", as_version=4); nbformat.validate(notebook); [ast.parse(cell.source, filename=f"sesion_06.ipynb:cell-{index}") for index, cell in enumerate(notebook.cells, start=1) if cell.cell_type == "code"]; packages = {"Python": platform.python_version(), "openai": version("openai"), "openai-agents": version("openai-agents"), "mcp": version("mcp"), "pandas": version("pandas"), "pydantic": version("pydantic"), "python-dotenv": version("python-dotenv"), "graphviz": version("graphviz"), "tiktoken": version("tiktoken"), "ipykernel": version("ipykernel"), "nbformat": version("nbformat")}; [print(f"{name}: {pkg_version}") for name, pkg_version in packages.items()]; print("Notebook: sesion_06.ipynb válido")'

notebook:
	@$(UV) run --frozen --python $(PYTHON_VERSION) jupyter lab $(NOTEBOOK)

clean:
	@rm -rf "$(VENV)" __pycache__ mcp_servers/__pycache__ "$(MPLCONFIGDIR)" "$(XDG_CACHE_HOME)"
