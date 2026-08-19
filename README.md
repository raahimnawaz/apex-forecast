# apex-forecast

[![CI](https://github.com/raahimnawaz/apex-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/raahimnawaz/apex-forecast/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

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
| 2a — Pit strategy | Minimum-time stop plan per car from measured degradation and pit loss (Bekker & Lotz 2009). Built and scored; **measured as no improvement, so not shipped** |
| 3 — Calibration | Walk-forward backtest, ranked probability score, benchmarked against three fitted baselines |
| Reliability | Hierarchical retirement hazard and a safety-car process, so the forecast is unconditional |

## How the model actually works

No neural network, and that is a decision rather than an omission. With ten in-regulation
races there are a few hundred usable finishing positions — nowhere near enough to learn
structure from scratch, and more than enough to overfit spectacularly. So the structure is
written down and only its parameters are learned. Everything below is fitted by MCMC, and
every claim it makes comes with an interval.

### The generative story

The model does not predict a finishing position. It describes **how a finishing order is
produced**, and then asks which parameters make the orders we actually saw most likely.

Give every car a single number, `θ` — its strength on the day. Then build a finishing order
by drawing cars one at a time, each with probability proportional to `exp(θ)`:

```
P(order) = Π_k   exp(θ_(k)) / Σ_{j ≥ k} exp(θ_(j))
```

That is the **Plackett-Luce** model — a softmax applied repeatedly, removing each car as it
is drawn. It is the natural extension of logistic regression from "which of two" to "in what
order", and it means a forecast is a distribution over *whole orderings*, not 22 independent
guesses. Joint questions therefore have real answers: P(Leclerc beats Verstappen) is a
quantity you can read off, not something assembled from marginals.

The strength itself is additive:

```
θ = driver skill + constructor strength   (+ grid effect, when qualifying has run)
```

### Why it runs backwards

The obvious version draws the *best* car first. It fails, and the failure is instructive.

Being classified 15th after first-lap damage is arithmetically identical to being slow, so
the forward model reads incidents as pace. On this data it rated **Ferrari's car above
Mercedes' while Mercedes won 8 of 11 races** — Leclerc's steady third and fourth places
looked like a fast car, and Mercedes' occasional disasters looked like a slow one.

So the shipped model draws the **worst** remaining car and reads the order in reverse: the
strongest cars are the ones that survive longest, and one bad result costs far less. That is
the attrition model of Graves, Reese & Fitzgerald (2003), proposed for NASCAR for exactly
this reason.

Measured at the moment of the switch, it moved the constructor share of explained spread
from **76% to 90%** — against 86–88% in the published literature — and lifted agreement with
the independent pace layer from **ρ 0.93 to 0.98**. Both have drifted slightly as races were
added; the current fit is reported under *What it found* below and on the dashboard, which
reads them from the model rather than from this file.

### Separating the driver from the car

This is the hard part, and it is *structurally* hard rather than merely difficult. Within a
fixed team-mate pairing you can add a constant to both drivers and subtract it from their
car with **no change in the likelihood at all** — the model cannot tell those apart. Three
things break the tie:

1. **Centring.** Driver and constructor vectors each have their mean subtracted, which pins
   the overall level.
2. **Drivers who changed teams.** The 19 who appear in both seasons are what tie the eras
   together; without them the 2026 constructor scale would float free.
3. **Hierarchical shrinkage.** Both sets are pulled toward zero, so extreme values need
   evidence to earn.

The intervals stay wide because the split genuinely is uncertain. **Read the intervals, not
the point estimates** — that is the honest output, not a hedge.

### Where the 2026 reset is encoded

Not in the data selection — in the **prior structure**:

- **Driver skill crosses the regulation boundary**, because skill is a property of the
  human. Each driver carries a 2025 level plus a partially-pooled 2026 delta, so the older
  season acts as a prior rather than as equally-weighted evidence. This is not decoration:
  forcing skill constant across both seasons put Antonelli — six wins and the championship
  lead — **seventh** in the forecast, because 24 mediocre rookie races outvoted 10 dominant
  ones.
- **Constructor strength is severed at it**, because new power units and aerodynamics mean
  2025 pace says nothing about 2026. The 2026 term is a fresh draw that then follows a
  **random walk** across rounds — that walk is how in-season development gets measured.
- Within 2026 driver skill is held fixed. With ten races and two cars per team, a walk on
  both skill and car is not identified; they would trade off freely race to race.

Fitted with NUTS in NumPyro, non-centred throughout — the funnel geometry of a hierarchical
model is severe otherwise, and the sampler quietly fails rather than loudly. R-hat is
checked on every run and reported on the page.

### Turning a posterior into a forecast

1. Push each constructor one step along its random walk, so uncertainty **grows with
   distance** to the race.
2. Sample orderings by the **Gumbel-max trick**: add standard Gumbel noise to each `θ` and
   sort. That is an exact draw from Plackett-Luce, not an approximation.
3. Draw retirements from a hierarchical logistic hazard, and a safety car from a
   per-circuit rate. Retired cars sort behind the finishers.
4. Count outcomes across draws to get the full 22×22 position matrix.

Retirements are deliberately **excluded from the strength likelihood** — a gearbox failure
is a reliability event, not a pace event, and letting it look like a slow car is exactly the
mistake the attrition model exists to avoid. They are modelled separately and recombined
here, which is what makes the published probabilities cover the whole field.

### How you know any of it is real

Two independent checks, and they are the reason to trust the rest:

- **Cross-validation between layers.** Layer 0 measures pace from lap times; Layer 1 infers
  strength from finishing orders. They share no likelihood, no data representation and no
  fitting method — so their agreement at ρ = 0.96 is evidence, not a model agreeing with
  itself.
- **Walk-forward backtesting.** For each round the model is refit on everything before it
  and asked to predict a race it has never seen. Baselines get the same treatment and are
  fitted distributions rather than point predictions, so "the grid predicts the finish" is a
  genuinely strong opponent. Results are published unchanged, including where the baseline
  still wins.

Every modelling choice in this repo was settled that way, including two that went against
the author's prior and one — pit strategy — that was built, measured, and left out.

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
| attrition+grid+strategy | 0.1028 | 0.0653 | 0.2624 | 0.4923 | 0.7548 |
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

Three variants are kept in the table because they lost. The forward Plackett-Luce reads an
incident-damaged 15th place as identical to being slow; the contamination model was the
textbook fix for that and forecast no better. The comparison is the evidence for the
variant that ships.

The third is the pit strategy layer, and it is the most interesting failure. It optimises
each car's stop plan against its own measured degradation and the circuit's measured pit
loss, and it forecasts **no better than the model without it** — RPS 0.1028 against 0.1026,
better in 2 of 7 races. The reason is worth knowing before rebuilding it: to stay
out-of-sample the target circuit must be given a *season-average* tyre profile, because its
own degradation is part of the race being predicted and Friday practice cannot substitute
(fuel burn-off and tyre age are collinear within a stint, so practice degradation fits with
the wrong sign). That removes most of the circuit specificity the layer existed to exploit.
What remains is 2–5% of the constructor spread, against a 7-race sample. `docs/HANDOFF.md`
carries the full post-mortem.

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
drivers started elsewhere after penalties — Hamilton forecast from P2 started P5, Antonelli
from P4 started P7.

That is now fixed properly rather than retrospectively. FastF1 leaves `GridPosition` empty
on a qualifying session and only fills it in on the race session, so the penalty-corrected
grid is available from no upstream source before the race — it is typed into
`grids/{season}_R{round}.csv`, which the build prefers over the classification. Falling
back now prints a loud warning instead of quietly getting six cars wrong. Across 2026 this
moves **16% of entries in 5 of 11 races**, and once 20 of 22 cars.

Every forecast is now also written to an immutable, timestamped prediction log
(`web/data/predictions/`) before the race, and `score_race.py` refuses to score anything
else — so this table cannot drift into grading a forecast that was revised after the fact.

## Status

Layers 0, 1 and 3 are built, plus a reliability and safety-car layer, so the forecast
covers the **whole field** rather than only the cars that finish. A pit strategy layer
(`src/apex/strategy.py`) is built, tested and scored, but does not ship — it was measured
and did not improve the forecast, so it stays a calibration variant rather than part of
the model.

The largest remaining gap is now **qualifying**: the forecast conditions on the starting
grid, so it cannot exist until Saturday evening.

**New here? Read [docs/HANDOFF.md](docs/HANDOFF.md).** It carries the design decisions and
their evidence, and — more usefully — the list of improvements that have already been
tried and measured as worse.
