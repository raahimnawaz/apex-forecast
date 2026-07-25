# apex-forecast — F1 Race Prediction Dashboard

Plan drafted 2026-07-24. Season state: **round 10 of 22 complete** (Belgian GP, 19 Jul).
Next race: **Hungarian GP, 26 Jul 2026**.

---

## 0. The one constraint that shapes everything

2026 is a **ground-up regulation reset**: new power units (350 kW MGU-K, ~8 MJ/lap, 50/50 ICE/electric,
sustainable fuel), active front+rear aero replacing DRS, driver-managed Overtake/Boost energy modes,
smaller/lighter cars, narrower tyres. Plus an **11-team, 22-car grid** — Cadillac debuts with zero
history, Audi rebrands Sauber.

Consequence: **pre-2026 car pace does not transfer.** Any model trained on 1950–2025 lap data and
pointed at Hungary 2026 is a fraud dressed as a model. What *does* transfer is **driver skill**
(a slowly-varying latent) and **circuit structure**. So the architecture is:

> weak, informative priors on driver skill carried from 2025 → a fast-adapting state-space model
> that relearns constructor pace from only 10 in-regulation races → wide, honest credible intervals.

The dashboard's job is to show that uncertainty, not hide it.

---

## 1. Data layer

| Source | Use | Notes |
|---|---|---|
| **FastF1 3.8.3** | laps, stints, tyre compound/age, car telemetry, weather, session results | Local disk cache. Pin the version. 3.8.3 fixed phantom laps for non-starters seen in the 2026 Chinese GP — spike-test a 2026 session ingest before building on it |
| **Jolpica-F1** (`api.jolpi.ca/ergast/f1/`) | historical results/standings 1950→now, Ergast-compatible schema | Ergast is dead; this is the successor. Rate-limited — cache hard, never hit it from the dashboard |
| **OpenF1** | live/session-time telemetry at 3.7 Hz, race control messages | Optional Phase 5 "live mode". Team radio coverage collapsed in 2026 — don't design around it |
| **Open-Meteo** | race-day weather forecast (no API key) | Rain flips the entire distribution; a forecast-free F1 model is a toy |

**Store:** idempotent ingest scripts → per-session Parquet → DuckDB views for feature assembly.
Everything reproducible from `make data`. Emit a data-quality report per session (missing laps,
deleted laps, red-flag segments, driver-code drift) — silent bad data is the #1 killer here.

---

## 2. Model stack

Four layers. Each has to earn inclusion by beating the layer below it out-of-sample.

### Layer 0 — Pace deconvolution (the highest-value piece)

Raw lap times are useless: they're fuel load + tyre age + traffic + track evolution + flags, with a
little bit of car pace buried inside. Fit a mixed-effects model per session:

```
laptime ~ driver + team + fuel_burn(lap) + compound × tyre_age (deg slope)
        + track_evolution(lap) + dirty_air(gap_ahead) + SC/VSC/red-flag + track_temp
```

with driver/team as random effects. Output: **fuel-and-tyre-corrected true pace per car**, plus
per-team **degradation slopes per compound**. This is motorsport engineering, not ML, and it's what
makes every downstream layer credible. `statsmodels` MixedLM to start, NumPyro if the hierarchy grows.

Quali gets its own model (low fuel, single lap, no traffic term).

### Layer 1 — Latent strength (Bayesian, time-varying)

**Multilevel rank-ordered logit / Plackett-Luce** with separate driver and constructor effects — the
published approach for disentangling driver skill from car advantage (constructor explains ~88% of
finishing-order variance in the hybrid era, so the split matters enormously).

2026-specific modifications:
- **Random-walk (state-space) prior** on each driver's and constructor's strength across rounds, so
  the model tracks in-season development instead of assuming a static season-long ability.
- **Asymmetric priors across the reg boundary:** driver skill inherits a moderately tight 2025
  posterior; constructor pace is reinitialised near-diffuse. Cadillac gets an explicit
  new-constructor prior (shrunk toward the back-of-grid mean, wide).
- Circuit-archetype interactions (power / high-downforce / street / mixed) rather than per-circuit
  effects — 10 races isn't enough for per-circuit terms.

Implementation: **NumPyro (JAX)** — fast on Apple Silicon CPU, NUTS, and the model is small enough
that full HMC is cheap. PyMC is the fallback if the JAX install fights back.

### Layer 2 — Monte Carlo race simulation

Marginal win probabilities are the easy part; the interesting quantities are *joint* (P(Leclerc
beats Verstappen), exact-podium combinations, points swings). Those require simulating the race.

Lap-by-lap discrete-event sim, ~10–50k runs per GP:
- grid from the quali model (with a start/turn-1 chaos model — position change distribution off the line)
- per-lap pace drawn from the Layer-0/1 joint posterior (correlated draws, not independent noise)
- tyre degradation + thermal window per compound per team
- pit-stop strategy: optimise each car's stop laps against its own deg curve, plus stop-time variance
- **overtaking model**: P(pass | pace delta, circuit overtaking difficulty, energy state). 2026-specific:
  the attacker's Overtake Mode budget and the defender's Boost — energy management is now a first-class
  state variable, which is a genuinely new modelling problem this season
- SC/VSC as a per-circuit hazard process, with the strategy layer reacting to it (free pit stops)
- DNF as a survival/hazard model per driver–team, with an elevated reliability hazard for year-one PUs

Output: full 22×22 finishing-position probability matrix, per-driver points distribution, championship
projection over the remaining 12 rounds.

### Layer 3 — Calibration & honest evaluation (non-negotiable)

Walk-forward backtest: train on rounds 1..k, predict k+1, for k = 4..9. Report:
- **Ranked Probability Score** (the right metric for ordered outcomes)
- log-loss on win / podium / points-finish
- Spearman ρ on predicted vs actual order
- **reliability diagrams** — when the model says 30%, does it happen 30% of the time?

Must beat these baselines, published side-by-side:
1. grid position = finish position
2. previous race's result
3. current championship order
4. de-vigged bookmaker implied probabilities *(benchmark only — this is a modelling project, not a betting tool)*

### Layer 4 — Gated ML residual model *(optional)*

LightGBM on the *residuals* of the structural model: circuit geometry features, telemetry-derived
straight-line-vs-cornering signature, weather, driver–circuit history. **Ships only if it beats the
structural model on the walk-forward split.** No black box bolted on for its own sake.

---

## 3. Dashboard

### Architecture

**Python pipeline → versioned JSON artifacts → static dark dashboard → GitHub Pages.**

Predictions change a few times per race weekend, not per user click, so a server buys nothing.
A precomputed static site is faster, free to host, trivially cacheable, and gives a public URL.
(FastAPI can be added later behind a `live/` route if OpenF1 in-session updating is wanted.)

No Node currently installed on this machine — this stack doesn't need it. If a component framework
is preferred later, Node is one `brew install` away, but vanilla + ES modules keeps it deployable today.

### Design language — dark, clean, elegant

- Background near-black, not pure black (`#0B0D10`); surfaces built from 3–4 elevation steps
  rather than shadows; hairline `1px` borders at low opacity.
- One accent colour for interactive/primary state. **Team colours used only for driver identity**
  (chips, avatars) — never as the categorical chart palette. F1's 2026 palette has multiple
  near-identical blues/reds and fails contrast; charts use an accessibility-validated palette instead.
- **Tabular/monospaced numerals for all timing data.** Lap times that jitter as digits change is the
  single most common tell of an amateur motorsport UI.
- Type: one technical sans (Inter / IBM Plex Sans), tight scale, generous line-height, 12-col grid.
- Motion: minimal, `prefers-reduced-motion` respected, no gratuitous transitions.
- Design references to pull from during the build: the `dataviz` skill (palette formula, chart mark
  specs, stat-tile and legend rules) and the `artifact-design` skill for page craft.

### Views

1. **Race Card** — next GP, session countdown, win-probability bars **with credible intervals**,
   weather forecast strip.
2. **Finishing-position heatmap** — 22 drivers × 22 positions probability matrix. The signature viz;
   it shows the *shape* of the uncertainty, not a single ranked list.
3. **Pace panel** — Layer-0 fuel/tyre-corrected pace ranking with error bars; degradation curves per
   compound per team; quali vs race pace divergence.
4. **Strategy explorer** — simulated optimal stop windows, undercut/overcut deltas, SC-timing sensitivity.
5. **Championship projection** — fan chart of title probability across the remaining 12 rounds.
6. **Model honesty page** — calibration curves, RPS vs the four baselines, and a plainly worded list
   of what the model does *not* know. This is the page that makes the project credible to an engineer
   reading it.

### The credibility feature

An **immutable, timestamped pre-race prediction log**, committed before every race and rendered as a
public track record. Anyone can check whether the model was right. Almost no F1 ML repo does this,
and it converts the project from "a notebook" into "a forecasting system."

---

## 4. Phases

| Phase | Work | Deliverable | Est. |
|---|---|---|---|
| 0 | uv + Python 3.12 scaffold, FastF1 cache, ingest 2026 R1–R10 + 2024–25 history | Parquet store + data-quality report | ½ day |
| 1 | Layer-0 pace deconvolution, feature tables | True-pace table + deg curves, sanity-checked against known race narratives | 1–2 d |
| 2 | Layer-1 NumPyro strength model + walk-forward backtest | Metrics table vs 4 baselines | 2–3 d |
| 3 | Layer-2 Monte Carlo sim (pit, overtake, SC, DNF) | Full outcome distributions + calibration report | 2–3 d |
| 4 | Dashboard: design system + 6 views, static build | Deployed GitHub Pages site | 2–3 d |
| 5 | GitHub Action: re-run post-session, regenerate JSON, redeploy, append prediction log | Self-updating system | 1 d |

~2 weeks part-time. **Recommended: cut a vertical slice first** — Phases 0→1→(crude 2)→4 with one
view, deployed, before deepening the model. A deployed thin thing beats an undeployed deep thing.

Realistic first target: **live predictions for the Hungarian GP is 2 days out and not achievable at
quality.** Aim the first real forecast at the round after, and backtest Hungary once it's run.

---

## 5. Risks, stated up front

- **10 races of in-regulation data.** Posteriors will be wide. The correct response is to *display*
  that width, not to tighten it with tricks. If the model looks confident about P5–P12, it's lying.
- **Cadillac has no history at all** — flag it in the UI as a high-uncertainty entry rather than
  pretending the estimate is comparable to Mercedes'.
- **FastF1 3.8.3 predates most of the 2026 season.** Spike-test ingest on one 2026 session in Phase 0
  before the pipeline is built on top of it; budget for schema quirks.
- **Jolpica is volunteer-run and rate-limited** (~$45/mo hosting, fundraising to break even). Cache
  everything locally; never make it a runtime dependency of the dashboard.
- Sprint weekends have a different session structure — handle explicitly, don't let them silently
  corrupt the practice-pace features.

---

## Sources

- FastF1 — https://docs.fastf1.dev/ · https://github.com/theOehrly/Fast-F1/releases
- Jolpica-F1 — https://github.com/jolpica/jolpica-f1
- OpenF1 — https://openf1.org/docs/
- Bayesian multilevel rank-ordered logit for F1 — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10660124/
- Truncated & time-weighted Plackett-Luce for F1 forecasting — https://projecteuclid.org/journals/bayesian-analysis/volume-13/issue-2/A-Comparison-of-Truncated-and-Time-Weighted-PlackettLuce-Models-for/10.1214/17-BA1048.pdf
- 2026 regulations — https://www.formula1.com/en/latest/article/the-beginners-guide-to-the-2026-regulations.6j0tS0hrHG2T01tpmK6XYz
