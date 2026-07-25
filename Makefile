PY := .venv/bin/python
SEASON ?= 2026

.PHONY: setup spike data pace web serve test lint all

setup:
	uv venv --python python3.12
	uv pip install -e ".[dev]"

spike:            ## verify FastF1 still parses a current-season session
	$(PY) scripts/spike_fastf1.py

data:             ## ingest sessions -> parquet + data-quality report
	$(PY) scripts/ingest.py --season $(SEASON)

pace:             ## Layer 0 deconvolution -> pace + degradation tables
	$(PY) scripts/build_pace.py --season $(SEASON)

web:              ## serialise model output for the dashboard
	$(PY) scripts/export_web.py --season $(SEASON)

serve:
	$(PY) -m http.server 8731 --directory web

test:
	$(PY) -m pytest tests/ -q

lint:
	.venv/bin/ruff check src scripts tests

all: data pace web
