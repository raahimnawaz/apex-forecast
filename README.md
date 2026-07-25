# apex-forecast

Probabilistic Formula 1 race forecasting for the 2026 regulation era, with a dark analytical dashboard.

2026 is a ground-up regulation reset — new power units (350 kW MGU-K, ~8 MJ/lap, 50/50 ICE/electric),
active aero replacing DRS, driver-managed Overtake/Boost energy modes, an 11-team grid. Pre-2026 car
pace does not transfer. This project treats that as the central modelling constraint rather than
training on decades of irrelevant history: driver skill carries a prior from 2025, constructor pace is
relearned from in-regulation races only, and the resulting uncertainty is displayed rather than hidden.

## Layers

| Layer | What it does |
|---|---|
| 0 — Pace deconvolution | Strips fuel load, tyre age, traffic and track evolution out of raw lap times to recover true car pace and per-team degradation slopes |
| 1 — Latent strength | Bayesian rank-ordered logit separating driver skill from constructor advantage, with a random-walk prior across rounds |
| 2 — Race simulation | Lap-by-lap Monte Carlo: starts, pit strategy, overtaking under energy constraints, safety cars, DNF hazard |
| 3 — Calibration | Walk-forward backtest, ranked probability score, reliability diagrams, benchmarked against four naive baselines |

## Data

- [FastF1](https://docs.fastf1.dev/) — laps, stints, tyre compound/age, telemetry, weather
- [Jolpica-F1](https://github.com/jolpica/jolpica-f1) — historical results and standings (Ergast successor)
- [Open-Meteo](https://open-meteo.com/) — race-day weather forecast

## Setup

```bash
uv venv --python python3.12
uv pip install -e .
```

## Usage

```bash
python scripts/spike_fastf1.py      # verify FastF1 against a 2026 session
python scripts/ingest.py --season 2026
python scripts/build_pace.py
python scripts/export_web.py
```

## Status

Phase 0–1: data pipeline and pace deconvolution.
