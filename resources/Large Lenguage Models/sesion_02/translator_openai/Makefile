SHELL := /bin/bash

PYTHON ?= python3.11
VENV := .venv
BIN := $(VENV)/bin
PYTHON_BIN := $(BIN)/python
PIP := $(PYTHON_BIN) -m pip
STREAMLIT := $(PYTHON_BIN) -m streamlit

.PHONY: help setup sync check run clean

help:
	@echo "Comandos disponibles:"
	@echo "  make setup  Crea el entorno virtual e instala dependencias"
	@echo "  make run    Ejecuta la app de Streamlit"
	@echo "  make check  Verifica imports y sintaxis"
	@echo "  make clean  Elimina el entorno virtual"

setup: $(PYTHON_BIN) sync check
	@echo "Entorno listo. Ejecuta: make run"

$(PYTHON_BIN):
	@$(PYTHON) -m venv $(VENV)

sync: $(PYTHON_BIN)
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt

check: $(PYTHON_BIN)
	@$(PYTHON_BIN) -m compileall -q app.py core
	@$(PYTHON_BIN) -c 'import openai, dotenv, streamlit; print("Imports OK")'

run: $(PYTHON_BIN)
	@$(STREAMLIT) run app.py

clean:
	@rm -rf $(VENV)

