# apex-forecast — handoff

Context for picking this up cold. Written 2026-07-26, after round 11 (Hungarian GP).

Read this before changing the model. Several of the obvious improvements have already
been tried and measured, and two of them made things worse.

---

## What this is

A probabilistic Formula 1 race forecaster for the 2026 regulation era, plus a dark
static dashboard. Two independent statistical layers, a reliability layer, and an
out-of-sample calibration harness. No chart library, no build step, no server.

The organising constraint: **2026 reset the regulations** — new power units, active
aero, 11 teams, 22 cars. Pre-2026 car pace does not transfer. Driver skill does. Most
of the model structure exists to respect that asymmetry.

---

## Current state

| | |
|---|---|
| Season | 2026, 11 of 22 rounds complete |
| Next race | R12 Dutch GP, 2026-08-23 (sprint weekend) |
| Data | 32,248 laps, 24 drivers, 11 constructors, 29 circuits |
| Training set | 39 races (24 in 2025, 15 in 2026 including 4 sprints) |
| Tests | 39 passing |
| Lint | ruff clean |

**Measured performance** — walk-forward, rounds 5–11, refitting before each race:

| model | RPS ↓ | ll_win ↓ | ll_podium ↓ | ll_points ↓ | ρ ↑ |
|---|---|---|---|---|---|
| **attrition+grid** (shipped) | **0.1026** | **0.0650** | 0.2625 | **0.4914** | 0.7546 |
| baseline: grid | 0.1095 | 0.0817 | **0.2532** | 0.9289 | **0.7628** |
| contaminated+grid | 0.1139 | 0.0842 | 0.3106 | 0.5843 | 0.7086 |
| forward+grid | 0.1144 | 0.0856 | 0.3088 | 0.5994 | 0.7090 |
| baseline: standings | 0.1330 | 0.1381 | 0.3241 | 1.0122 | 0.6733 |
| baseline: last race | 0.1441 | 0.1591 | 0.3798 | 0.8491 | 0.5261 |

It wins **3 of 5 metrics**. The grid baseline still orders the field slightly better
(podium log-loss, Spearman). Paired margin **t = 1.48 against a 2.45 threshold**, winning
4 of 7 races — **promising, not established.** The page says exactly this. Do not
upgrade that claim without the numbers moving.

**Live test, round 11** (forecast committed before the race existed): called the winner
(Norris, 68.5%), RPS 0.0762 vs 0.0821 for a fitted actual-grid baseline on classified
finishers, 0.0938 vs 0.1086 on the full field. Podium 1/3.

---

## Architecture

```
FastF1 + Jolpica ──> data/raw/*.parquet ──> data/processed/*.parquet
                                                │
      ┌─────────────────────────────────────────┼──────────────────────┐
      │                                         │                      │
  Layer 0                                   Layer 1              reliability
  pace.py                                  strength.py          reliability.py
  Huber regression per race           Bayesian rank-ordered      hierarchical
  strips fuel/tyre/traffic            logit, NumPyro/NUTS        logistic DNF
      │                                         │                      │
      └──────────────> cross-check ρ=0.96 <─────┘                      │
                                                │                      │
                                          simulate_race() <────────────┘
                                                │
                                    web/data/*.json ──> static dashboard
```

- **`src/apex/pace.py`** — Layer 0. One robust (Huber) regression per race:
  `lap_time ~ C(Driver) + C(Compound) + C(Compound):tyre_age + lap_c + dirty_air`.
  Outputs fuel/tyre/traffic-corrected pace and per-compound degradation.
- **`src/apex/strength.py`** — Layer 1. Plackett-Luce family over finishing orders.
  Three likelihoods implemented; **attrition ships** (see below).
- **`src/apex/reliability.py`** — retirement hazard + safety-car process +
  `simulate_race()`, which produces the full-field distribution.
- **`src/apex/teams.py` / `news.py` / `weather.py` / `trackart.py`** — dashboard inputs.
- **`scripts/calibrate.py`** — walk-forward bake-off. **The arbiter.** Model choices are
  settled here, not by argument.
- **`scripts/score_race.py`** — scores an already-published forecast against a race that
  has since run. The one test that cannot be gamed.

---

## Decisions that will look wrong until you know why

**1. The rank model runs backwards (attrition / reverse Plackett-Luce).**
Drawing the *worst* remaining car, not the best. The forward model reads an
incident-damaged 15th place as identical to being slow. Three such races rated Mercedes'
car below Ferrari's despite eight wins. Switching to attrition moved the constructor
share 76% → 90% (published values are 86–88%) and Layer 0 agreement ρ 0.93 → 0.98.
Source: Graves, Reese & Fitzgerald (2003), via Henderson & Kirrane (2018).

**2. Only the top 10 are modelled as an ordering.**
Truncated Plackett-Luce. Positions outside the points still populate every denominator
but are not read as a precise ranking. Henderson & Kirrane motivate the same cut.

**3. Retirements are excluded from the rank likelihood, and modelled separately.**
A DNF is a reliability event, not a pace event. van Kesteren & Bergkamp make the same
choice. Consequence: Layer 1 skill excludes reliability by construction.

**4. Driver skill crosses the regulation boundary; constructor strength does not.**
Each driver has a 2025 level plus a partially-pooled 2026 delta. Forcing skill constant
across both seasons put Antonelli — six wins, championship leader — **seventh** in the
forecast, because 24 mediocre rookie races outvoted 10 dominant ones.

**5. Driver skill is fixed *within* 2026; only the car follows a random walk.**
With 10 races and two cars per team, a walk on both is not identified.

**6. Two models are fitted, not one.** Grid-free for the driver-vs-car split;
grid-conditional for the race forecast. Conditioning on qualifying absorbs part of the
effect the unconditional model is trying to measure. They answer different questions.

**7. Team colours never encode a value.** On the dark surface the 2026 constructor
palette fails colour-vision checks outright — Haas silver vs Mercedes teal is ΔE 0.6
under deuteranopia; Alpine vs Red Bull is ΔE 9.2 under *normal* vision. Team colour is a
chip beside a text label; every quantitative scale uses a validated ramp.

---

## Tried, measured, rejected — do not redo these

| Idea | Result |
|---|---|
| **Forward Plackett-Luce** | RPS 0.1144 — loses to the grid baseline. Reads incidents as pace. |
| **Contamination model** (uniform noise mixed into each selection step) | Estimated a plausible ε ≈ 5.5–6% and forecast **no better** than the model it was meant to repair (0.1139 vs 0.1144). The textbook robust answer does not work here. |
| **Correlated retirements** ("races retire cars in clumps") | **The premise is false.** 6 of 41 timed retirements fall within a lap of another; a permutation test expects 5.5 (sd 3.0), p = 0.52. Only 3 retirements in the first three laps; median retirement is lap 24. There is no clustering to model. |
| **Constant driver skill across 2025–26** | Put the championship leader 7th. |
| **Counting "Did not start" as a retirement** | Wrong twice — inflates the denominator with a start that never happened and the numerator with a non-race failure. Was concentrated in McLaren (3 of 7). Piastri's real record is 1 of 9, not 3 of 11. |

---

## Known limitations, ranked

1. **No pit strategy.** Degradation curves are computed and displayed but the forecast
   never uses them. ~2 stops per driver per race and 13.5 laps of spread in stop timing
   within a single race — real, measurable, ignored. **This is the biggest remaining gap.**
2. **Safety cars are reduced-form.** A safety car raises the entropy of the finishing
   order rather than being simulated through the pit lane. Stand-in, not mechanism.
3. **Calibration sample is 7 races.** t = 1.48. Not established.
4. **Driver/car split is weakly identified.** Two cars per team means adding a constant
   to both team-mates and subtracting from the car changes nothing. Only the 19 drivers
   present in both seasons break the tie. Read intervals, not point estimates.
5. **11,657 practice laps and 4,418 qualifying laps ingested and unused.** Layer 0 only
   fits race sessions. Friday long runs are the strongest pre-race signal available.
6. **No quali model.** Before qualifying runs there is no grid to condition on.
7. **Wet regime not fitted.** One wet race in 2026. Weather currently only widens the
   forecast as a stated assumption.
8. **Backmarkers flattered less than they deserve** — blue-flag lifting is only partly
   absorbed by the dirty-air term.

---

## Running it

```bash
make setup                    # uv venv on python 3.12 + deps
make spike                    # verify FastF1 still parses a current-season session
make all                      # ingest -> pace -> strength -> web payloads -> news
make serve                    # http://localhost:8731
make test lint
make calibrate                # slow: refits per round
make trackart ROUND=11        # hero artwork from the pole lap
python scripts/score_race.py --round 11   # score a published forecast after the fact
```

`data/` is gitignored and fully reproducible — a fresh clone needs `make data` before
anything else. `web/data/*.json` **is** committed so the dashboard is self-contained.

---

## Environment gotchas

- **Python 3.12 only.** System python3.14 breaks the scientific stack. The venv is `.venv`.
- **No Node**, and none needed — the dashboard is hand-rolled SVG with ES modules.
- **FastF1 is pinned to 3.8.3**, which predates most of the 2026 season. It works, but
  run `make spike` after upgrading anything.
- **Jolpica is volunteer-run and rate-limited.** Everything is cached to `data/raw`.
  Never call it at view time.
- **Bump `?v=` on `style.css` / `app.js` / `hero.js` in both HTML files** when assets
  change, or a stale cache will serve the old ones. This has bitten twice.
- **Liquid glass needs Chromium.** SVG filters in `backdrop-filter` are Chromium-only;
  everything else gets a frosted fallback. Feature-detected, never sniffed.
- **Don't mount visual effects from `requestAnimationFrame` or `ResizeObserver` alone** —
  neither fires reliably in embedded preview contexts. Build synchronously, observe after.
- **CSS specificity trap:** `.glass` sets `position: relative` and appears late in the
  stylesheet. Anything positioning a glass element needs a more specific selector.
- **Screenshots in the preview pane are unreliable.** Verify layout by measuring the DOM
  (`getBoundingClientRect`, computed styles), not by looking at a capture.

---

## If you change the model

Run `make calibrate` and let the numbers decide. Every model choice in this repo was
settled that way, including two that went against my prior. Publish the result whatever
it says — the dashboard already renders a losing verdict correctly, and the significance
test is computed from the data rather than written into the copy.

Do not upgrade "promising" to "proven" until t clears the threshold at the actual
sample size.
