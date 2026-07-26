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
| 3 — Calibration | Walk-forward backtest, ranked probability score, benchmarked against three fitted baselines |
| Reliability | Hierarchical retirement hazard and a safety-car process, so the forecast is unconditional |

## What it found

- **Corrected pace and finishing-order strength agree at Spearman ρ = 0.96** across the
  eleven constructors. The two layers share no data representation, likelihood or
  fitting method, so the agreement is evidence rather than a self-consistency check.
- **93% of the finishing-order spread is the car**, in line with the 88% and 86%
  reported in the literature for the hybrid era.
- **Monaco costs 2.53 s per lap in dirty air** against 0.14–0.83 s everywhere else. The
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

## Calibration

Walk-forward over rounds 5–11, refitting from scratch before each race. Baselines are
fitted Plackett-Luce distributions rather than point predictions, so "starting position
predicts the finish" competes on equal terms.

| model | RPS ↓ | ll win ↓ | ll podium ↓ | ll points ↓ | ρ ↑ |
|---|---|---|---|---|---|
| **attrition+grid** (shipped) | **0.1026** | **0.0650** | 0.2625 | **0.4914** | 0.7546 |
| baseline: grid | 0.1095 | 0.0817 | **0.2532** | 0.9289 | **0.7628** |
| contaminated+grid | 0.1139 | 0.0842 | 0.3106 | 0.5843 | 0.7086 |
| forward+grid | 0.1144 | 0.0856 | 0.3088 | 0.5994 | 0.7090 |
| baseline: standings | 0.1330 | 0.1381 | 0.3241 | 1.0122 | 0.6733 |
| baseline: last race | 0.1441 | 0.1591 | 0.3798 | 0.8491 | 0.5261 |

Reading it honestly: the model takes **3 of 5 metrics**, and the grid baseline still
orders the field slightly better on podium log-loss and Spearman. The paired margin over
the baseline is **t = 1.48 against a 2.45 threshold**, winning 4 of 7 races — promising,
not established. The dashboard says exactly that, and computes the test from the data
rather than asserting it in copy.

Where the model is clearly ahead is the midfield: points log-loss **0.49 against 0.93**.
The grid baseline is badly overconfident about who scores.

Two variants are kept in the table because they lost. The forward Plackett-Luce reads an
incident-damaged 15th place as identical to being slow; the contamination model was the
textbook fix for that and forecast no better. The comparison is the evidence for the
variant that ships.

## Live test — Hungarian Grand Prix, round 11

The forecast was committed before the race existed, then scored by
`scripts/score_race.py`:

| scored | model | RPS ↓ | ll win ↓ | ll points ↓ |
|---|---|---|---|---|
| classified | **forecast** | **0.0762** | **0.0375** | **0.2250** |
| classified | baseline: actual grid | 0.0821 | 0.0499 | 0.3182 |
| full field | **forecast** | **0.0938** | **0.0328** | **0.3092** |
| full field | baseline: actual grid | 0.1086 | 0.0491 | 0.4853 |

It called the winner (Norris, 68.5% from pole) and got the podium 1 of 3. It also made
one avoidable error: the forecast used the qualifying classification as the grid, and six
drivers started elsewhere after penalties. The build now prefers the actual starting grid.

## Status

Layers 0, 1 and 3 are built, plus a reliability and safety-car layer, so the forecast now
covers the **whole field** rather than only the cars that finish. Pit strategy is the
largest remaining gap: degradation curves are computed and displayed but the forecast does
not use them.

**New here? Read [docs/HANDOFF.md](docs/HANDOFF.md).** It carries the design decisions and
their evidence, and — more usefully — the list of improvements that have already been
tried and measured as worse.
