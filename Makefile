SHELL := /bin/bash

PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin
PYTHON_BIN := $(BIN)/python
PIP := $(PYTHON_BIN) -m pip
STREAMLIT := $(PYTHON_BIN) -m streamlit
NOTEBOOK := notebooks/notebook_asistente_tenerife.ipynb

.PHONY: help setup sync check run test eval notebook clean

help:
	@echo "Comandos disponibles:"
	@echo "  make setup     Crea el entorno virtual e instala dependencias"
	@echo "  make run       Ejecuta la app de Streamlit"
	@echo "  make test      Lanza la suite de pytest"
	@echo "  make eval      Ejecuta la evaluación (LLM-as-judge) y genera el informe"
	@echo "  make notebook  Abre JupyterLab con el notebook entregable"
	@echo "  make check     Verifica imports y sintaxis"
	@echo "  make clean     Elimina el entorno virtual y las cachés locales"

setup: $(PYTHON_BIN) sync check
	@echo "Entorno listo. Copia .env.template a .env, añade GOOGLE_API_KEY y ejecuta: make run"

$(PYTHON_BIN):
	@$(PYTHON) -m venv $(VENV)

sync: $(PYTHON_BIN)
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt

check: $(PYTHON_BIN)
	@$(PYTHON_BIN) -m compileall -q app.py core scripts
	@$(PYTHON_BIN) -c 'import streamlit, langchain, langchain_core, langchain_community, langchain_google_genai, langchain_text_splitters, faiss, fitz, dotenv; print("Imports OK")'

run: $(PYTHON_BIN)
	@$(STREAMLIT) run app.py

test: $(PYTHON_BIN)
	@$(PYTHON_BIN) -m pytest

eval: $(PYTHON_BIN)
	@$(PYTHON_BIN) -m scripts.run_eval

notebook: $(PYTHON_BIN)
	@$(BIN)/jupyter lab $(NOTEBOOK)

clean:
	@rm -rf $(VENV) __pycache__ core/__pycache__ tests/__pycache__ scripts/__pycache__
