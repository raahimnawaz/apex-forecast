PY := .venv/bin/python
SEASON ?= 2026

.PHONY: setup spike status data pace strength calibrate web news serve test lint all

status:           ## where the season is and the one thing to do next
	@$(PY) scripts/status.py --season $(SEASON)

setup:
	uv venv --python python3.12
	uv pip install -e ".[dev,model]"

spike:            ## verify FastF1 still parses a current-season session
	$(PY) scripts/spike_fastf1.py

data:             ## ingest sessions -> parquet + data-quality report
	$(PY) scripts/ingest.py --season $(SEASON)

pace:             ## Layer 0 deconvolution -> pace + degradation tables
	$(PY) scripts/build_pace.py --season $(SEASON)

strength:         ## Layer 1 rank-ordered logit -> skill, car strength, forecast
	$(PY) scripts/build_strength.py

calibrate:        ## walk-forward backtest against naive baselines (slow: refits per round)
	$(PY) scripts/calibrate.py

web:              ## serialise model output for the dashboard
	$(PY) scripts/export_web.py --season $(SEASON)

trackart:         ## regenerate the hero artwork (needs ROUND=n)
	$(PY) scripts/build_trackart.py --season $(SEASON) --round $(ROUND)

news:             ## refresh headlines only (seconds; no model refit needed)
	$(PY) scripts/export_news.py

serve:
	$(PY) -m http.server 8731 --directory web

test:
	$(PY) -m pytest tests/ -q

lint:
	.venv/bin/ruff check src scripts tests

all: data pace strength web news
