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

## What it found

- **Corrected pace and finishing-order strength agree at Spearman ρ = 0.93** across the
  eleven constructors. The two layers share no data representation, likelihood or
  fitting method, so the agreement is evidence rather than a self-consistency check.
- **76% of the finishing-order spread is the car**, the rest the driver — lower than the
  ~88% reported for the settled hybrid era, which is what a regulation reset should do.
- **Monaco costs 2.53 s per lap in dirty air** against 0.14–0.76 s everywhere else. The
  model was not told Monaco is different; it recovered that from lap times.

## Data

- [FastF1](https://docs.fastf1.dev/) — laps, stints, tyre compound/age, telemetry, weather
- [Jolpica-F1](https://github.com/jolpica/jolpica-f1) — historical results (Ergast successor)
- Publisher RSS feeds — headlines, shown verbatim with attribution and a source link

## Setup

```bash
make setup
```

## Usage

```bash
make spike     # verify FastF1 still parses a current-season session
make all       # ingest -> pace -> strength -> web payloads -> news
make serve     # dashboard at http://localhost:8731
make test lint
```

`make news` refreshes headlines on its own — it needs no fitted posterior and runs in
seconds, so the news section can update on a different schedule to the models.

## Status

Layers 0 and 1 are built and cross-validated. Layer 2 (race simulation with pit strategy,
overtaking under energy constraints, safety cars and a DNF hazard model) and Layer 3
(walk-forward calibration against naive baselines) are not built yet — so the published
forecast is **conditional on finishing** and is labelled that way on the page.
