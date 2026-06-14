PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin
PIP := $(BIN)/pip
STREAMLIT := $(BIN)/streamlit

.PHONY: setup run check clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(STREAMLIT) run app.py

check:
	$(BIN)/python -m compileall app.py core

clean:
	rm -rf $(VENV)
