# MTA Transit Line Predictor — Makefile
# Python backend (FastAPI + ML) + TypeScript frontend
# Quickstart: `make setup` then `make dev`

# --- Python ---
PYTHON := python3
VENV := .venv
BIN := $(VENV)/bin
PIP := $(BIN)/pip
PY := $(BIN)/python

# --- Frontend ---
FRONTEND_DIR := frontend
NPM := npm

# --- Data ---
GTFS_URL := https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip
RIDERSHIP_URL := https://data.ny.gov/api/views/wujg-7c2s/rows.csv?accessType=DOWNLOAD
DATA_DIR := data
RAW_DIR := $(DATA_DIR)/raw
PROCESSED_DIR := $(DATA_DIR)/processed
MODELS_DIR := models

.DEFAULT_GOAL := help
.PHONY: help setup setup-py setup-fe venv install dirs env \
        data gtfs ridership preprocess train \
        dev backend frontend jupyter \
        freeze clean nuke

help:  ## show targets
	@echo "MTA Transit Line Predictor"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
setup: setup-py setup-fe  ## one-shot: python + frontend

setup-py: venv install dirs env  ## python backend setup

setup-fe:  ## scaffold TS frontend (vite + react + leaflet)
	@if [ ! -d $(FRONTEND_DIR) ]; then \
		echo ">>> scaffolding vite + react + ts frontend"; \
		$(NPM) create vite@latest $(FRONTEND_DIR) -- --template react-ts; \
		cd $(FRONTEND_DIR) && $(NPM) install && \
		$(NPM) install leaflet react-leaflet @types/leaflet \
			leaflet-draw @types/leaflet-draw react-leaflet-draw \
			axios recharts; \
	else \
		echo ">>> frontend already exists, installing deps"; \
		cd $(FRONTEND_DIR) && $(NPM) install; \
	fi

venv: $(VENV)/bin/activate  ## create python venv

$(VENV)/bin/activate:
	@echo ">>> creating venv at $(VENV)"
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel

install: venv  ## install python requirements
	$(PIP) install -r requirements.txt

dirs:  ## create project structure
	mkdir -p $(RAW_DIR) $(PROCESSED_DIR) $(MODELS_DIR) src notebooks

env:  ## create .env template if missing
	@if [ ! -f .env ]; then \
		echo "NYC_OPEN_DATA_APP_TOKEN=" > .env; \
		echo "API_HOST=0.0.0.0" >> .env; \
		echo "API_PORT=8000" >> .env; \
		echo ">>> created .env template"; \
	fi

freeze:  ## pin python deps
	$(PIP) freeze > requirements.lock

# ----------------------------------------------------------------------------
# Data pipeline
# ----------------------------------------------------------------------------
data: gtfs ridership  ## download all raw data

gtfs: dirs  ## fetch MTA subway GTFS schedule
	@echo ">>> downloading MTA GTFS"
	curl -L -o $(RAW_DIR)/gtfs_subway.zip $(GTFS_URL)
	cd $(RAW_DIR) && unzip -o gtfs_subway.zip -d gtfs_subway

ridership: dirs  ## fetch MTA ridership CSV
	@echo ">>> downloading MTA ridership (large, may take a min)"
	curl -L -o $(RAW_DIR)/mta_ridership.csv "$(RIDERSHIP_URL)"

preprocess:  ## clean + feature-engineer
	$(PY) src/preprocess.py --raw $(RAW_DIR) --out $(PROCESSED_DIR)

train:  ## train ridership + cost models
	$(PY) src/train.py --data $(PROCESSED_DIR) --out $(MODELS_DIR)

# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
dev:  ## run backend + frontend together (needs 2 terminals or use `make backend` / `make frontend`)
	@echo "Run these in two terminals:"
	@echo "  make backend"
	@echo "  make frontend"

backend:  ## start FastAPI backend at :8000
	$(BIN)/uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

frontend:  ## start vite dev server (TS frontend)
	cd $(FRONTEND_DIR) && $(NPM) run dev

jupyter:  ## launch jupyter for exploration
	$(BIN)/jupyter notebook --notebook-dir=notebooks

# ----------------------------------------------------------------------------
# Cleanup
# ----------------------------------------------------------------------------
clean:  ## remove caches
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
	find . -type d -name node_modules -prune -exec rm -rf {} +

nuke: clean  ## clean + remove venv, data, models
	rm -rf $(VENV) $(DATA_DIR) $(MODELS_DIR) $(FRONTEND_DIR)