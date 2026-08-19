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
| attrition+grid+strategy | 0.1028 | 0.0653 | 0.2624 | 0.4923 | 0.7548 |
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
| **Pit strategy** (`src/apex/strategy.py`) | Built, leak-free, **and it does not help**: RPS 0.1028 against 0.1026 for the same model without it. Paired t = +2.09 (p = 0.082) in the *wrong* direction, better in 2 of 7 races. Kept as a scored variant, not shipped. See below for why. |
| **Tyre profile from Friday practice** | **Structurally impossible, not merely noisy.** Fitted degradation comes out *negative* — Spa −0.29 s/lap, Hungaroring −0.07, against race values of +0.06 and +0.08 — and +3.86 at a wet Canadian weekend. Within a long run the car burns fuel as the tyre ages, so fuel load and tyre age are both linear in lap-within-stint and perfectly collinear. A race escapes this only because drivers pit on different laps; practice has no shared clock. Separating them needs an assumed kg/lap coefficient, which `pace.py` refuses to invent. |
| **Forward Plackett-Luce** | RPS 0.1144 — loses to the grid baseline. Reads incidents as pace. |
| **Contamination model** (uniform noise mixed into each selection step) | Estimated a plausible ε ≈ 5.5–6% and forecast **no better** than the model it was meant to repair (0.1139 vs 0.1144). The textbook robust answer does not work here. |
| **Correlated retirements** ("races retire cars in clumps") | **The premise is false.** 6 of 41 timed retirements fall within a lap of another; a permutation test expects 5.5 (sd 3.0), p = 0.52. Only 3 retirements in the first three laps; median retirement is lap 24. There is no clustering to model. |
| **Constant driver skill across 2025–26** | Put the championship leader 7th. |
| **Counting "Did not start" as a retirement** | Wrong twice — inflates the denominator with a start that never happened and the numerator with a non-race failure. Was concentrated in McLaren (3 of 7). Piastri's real record is 1 of 9, not 3 of 11. |

---

## Why the pit strategy layer did not help

Worth reading before anyone builds it again, because the layer itself is sound — the
optimiser is exact and brute-force tested, degradation and pit loss are measured rather
than assumed, and the offsets are centred so they re-rank without shifting the field.

**The leak-free version has almost nothing left to say.** The optimiser needs to know how
a circuit treats tyres. For a race that has not happened, that cannot come from the race
itself, and practice cannot supply it (see the table above). So an unraced circuit gets a
*season-average* tyre profile — which means degradation no longer varies by circuit at
all, and the only circuit-specific inputs left are race distance, longest observed stint
and pit loss. Most of the signal was in the part that had to be removed.

What survives is small: offsets spread 0.10–0.29 theta against a constructor spread of
**5.97 theta**, so 2–5% of the effect the model already has. And Layer 1 is not blind to
strategy in the first place — it fits finishing orders that teams produced *while running
their actual strategies*, so the season-average part of a car's tyre weakness is already
in its strength term. The circuit-conditional remainder is what this layer adds, and at a
7-race sample it is smaller than the noise.

Before rebuilding it, the thing to fix is not the optimiser. It is the missing pre-race
tyre profile — which needs either a fuel-corrected practice model (and therefore a
defensible kg/lap coefficient) or enough seasons to learn circuit tyre character as a
prior. Neither exists yet.

---

## Known limitations, ranked

1. **No qualifying model.** The forecast conditions on the grid, so it cannot exist until
   Saturday evening. **This is now the biggest gap** — it is the difference between a
   forecasting system and a Saturday-night one.
2. **Safety cars are reduced-form.** A safety car raises the entropy of the finishing
   order rather than being simulated through the pit lane. Stand-in, not mechanism. Now
   that a strategy layer exists, the sharpest missing piece is the interaction: a safety
   car is a half-price pit stop, and that is the most common way a race is won from behind.
3. **Pit strategy is built but does not ship.** `strategy.py` is a scored calibration
   variant, not part of the forecast. Degradation is modelled as linear, so it understates
   the value of an extra stop on a compound that cliffs; there is no undercut, no traffic
   and no tyre warm-up, so it computes a free-air time optimum rather than a race.
4. **Calibration sample is 7 races.** t = 1.48. Not established.
5. **Driver/car split is weakly identified.** Two cars per team means adding a constant
   to both team-mates and subtracting from the car changes nothing. Only the 19 drivers
   present in both seasons break the tie. Read intervals, not point estimates.
6. **Practice laps remain unused, now for a stated reason.** Layer 0 fits race sessions
   only. Friday long runs look like the strongest pre-race signal available and are not,
   at least for tyres: fuel burn-off and tyre age are collinear within a stint, so the
   fitted degradation comes out with the wrong sign. See the rejected table.
7. **Wet regime not fitted.** One wet race in 2026. Weather currently only widens the
   forecast as a stated assumption.
8. **Backmarkers flattered less than they deserve** — blue-flag lifting is only partly
   absorbed by the dirty-air term.
9. **No championship projection.** Specified in `PLAN.md` §3, not built. (The prediction
   log, the other item from that section, now exists — see below.)
10. **Team form is circuit-dependent and the model does not know it.** A team's corrected
    pace varies by circuit far beyond noise — F = 2.68, p = 5e-07, interaction spread
    0.59 s/lap against a 0.21 s/lap team-mate noise floor, adjusted R² 0.860 → 0.923. It is
    the largest unexploited signal in the data. It is also **not currently forecastable**:
    every circuit is raced exactly once per season, so a team-circuit cell has one race of
    evidence and nothing transfers across the 2026 regulation boundary. Pooling into
    archetypes is the only route, and the archetype axes available today (dirty-air cost,
    degradation, lap length) buy +0.012 adjusted R² for 22 parameters — BIC rejects them
    outright. Real circuit geometry is the missing input, and it needs telemetry, which
    `ingest.py` currently loads with `telemetry=False`.
11. **Undated news items claim to be the newest.** `_parse_date` falls back to
    `datetime.now(UTC)` when a feed omits or malforms `pubDate`, and the list is sorted
    newest-first. Measured 2026-08-19: **10 of 60 items** carried a fabricated timestamp,
    all identical, so they occupied the entire top of the dashboard ahead of genuinely
    recent headlines — and re-dated themselves on every fetch, which is why the weekly
    refresh has to ignore `published` to tell a real change from a clock tick. The honest
    fix is to stop inventing a time: either carry forward the timestamp from the first
    fetch that saw the link, or sort unknown dates last instead of first. Both need a
    decision about what the dashboard should show for an item whose age is unknown.

---

## The prediction log

`web/data/predictions/{season}_R{round}.json`, written once per round by `export_web.py`
and **never overwritten**. This is the credibility feature from `PLAN.md` §3.

It exists because `strength_{season}.json` always holds the *next* race. Scoring round 11
after the round 12 build has run used to silently grade the wrong forecast and print
confident numbers for it. `score_race.py` now reads the log and refuses to score at all
rather than fall back to a payload belonging to another round.

Round 11 was backfilled from git blob `d52f263` (commit `6c7b122`), the forecast that was
actually published before the Hungarian Grand Prix. It reproduces the committed
`reports/race_score_2026_R11.csv` exactly — RPS 0.0762, ll_win 0.0375, ρ 0.8772 — which is
what confirms the backfill is the genuine article and not a refit.

---

## Running it

**Before forecasting a race that has not run, write `grids/{season}_R{round}.csv`.**
FastF1 leaves `GridPosition` empty on a qualifying session and only fills it in on the race
session, so the grid corrected for penalties is available from no upstream source this
project can reach — it is typed in from the stewards' decisions. Without it the build falls
back to the qualifying classification and says so loudly. That fallback is what cost the
round 11 forecast: Hamilton was forecast from P2 and started P5, Antonelli from P4 and
started P7. Across 2026 it moves 16% of entries in 5 of 11 races. See `grids/README.md`.

**Start with `make status`.** It reads the state off disk — last round ingested, whether
qualifying has run, whether a corrected grid exists, what has been forecast and what has
been scored — and prints the single next action. It fits nothing and writes nothing, so it
is safe to run at any time. It exists because the order of the other targets used to live
only in someone's head, and the expensive failure mode was silent.

```bash
make status                   # where the season is, and the one thing to do next
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
- **Bump `?v=` when assets change — in three places, not one.** This has now bitten three
  times, and the third time was because the documented fix was itself incomplete:
  1. the `<link>` and `<script>` tags in **both** HTML files;
  2. the **import specifiers inside modules** — `hero.js` imports `silk.js` and `glass.js`,
     `bg.js` imports `silk.js`. A module's imports are fetched as their own requests with
     their own cache entries, so `hero.js?v=N` busts hero.js alone and leaves a rewritten
     `silk.js` served from cache. The background can be replaced wholesale and render
     identically, which is as confusing as it sounds;
  3. bump **all of them in one edit**. Bumping the HTML, reloading, then editing the module
     leaves the fresh version number pointing at stale contents — and that gets cached too.
- **The dev server caches the HTML itself.** `python -m http.server` sends no cache-control,
  so the browser can reuse `index.html` and never see the new `?v=`. When checking a visual
  change locally, load `http://localhost:8731/?cb=<anything-new>` rather than trusting a
  reload. `read_network_requests` shows which `?v=` was actually fetched and settles it in
  seconds. GitHub Pages sends proper validators, so this one is local-only.
- **Liquid glass needs Chromium.** SVG filters in `backdrop-filter` are Chromium-only;
  everything else gets a frosted fallback. Feature-detected, never sniffed.
- **Don't mount visual effects from `requestAnimationFrame` or `ResizeObserver` alone** —
  neither fires reliably in embedded preview contexts. Build synchronously, observe after.
- **CSS specificity trap:** `.glass` sets `position: relative` and appears late in the
  stylesheet. Anything positioning a glass element needs a more specific selector.
- **Screenshots in the preview pane are unreliable.** Verify layout by measuring the DOM
  (`getBoundingClientRect`, computed styles), not by looking at a capture.

---

## How it stays current

The refresh runs on this Mac, not in GitHub Actions. `livetiming.formula1.com` answers
403 to datacenter IPs and FastF1's mirror does not carry archived sessions, so a hosted
runner cannot ingest a race weekend at all — measured, see the header of
`.github/workflows/update.yml`. CI there runs tests and lint only.

`ops/refresh.sh` does the whole job: pull, spike, `make all`, score the round that just
ran, test, lint, commit, push. Pushing `web/data` triggers `pages.yml`, so the live
dashboard redeploys itself. Two LaunchAgents run it, both installed from `ops/`:

| agent | when | what it is for |
|---|---|---|
| `com.raahimnawaz.apex-refresh` | Tuesday 09:00 | ingest the race that just ran, refit, score |
| `com.raahimnawaz.apex-qualifying` | Saturday 18:00 | catch the grid once qualifying has run |

The Saturday run exists because the prediction log is write-once and now defers while a
forecast is grid-free. Without it, a round can only ever be logged grid-free, which is
not the forecast this project ships and is not comparable to the rounds already scored.
F1 qualifies at very different local times, so the Saturday agent will sometimes fire
early; that is harmless, the log just defers again.

Guards worth knowing about, because each one exists for something that actually
happened:

- **Refuses to run on a dirty tree.** It commits on your behalf; it will not do that on
  top of a half-finished edit.
- **Stops quietly if the feed refuses the machine** (spike exit 2) rather than failing.
  A genuine parse failure (exit 1) still fails loudly.
- **Will not publish if tests or lint fail.**
- **Ignores `generated_utc` and `published` when deciding whether anything changed.**
  Every build restamps the clock, so without this it would commit and redeploy every
  week with nothing new. See limitation 11.

Logs are in `reports/refresh.log` (gitignored). That is the first place to look if a
round is missing from the calibration sample.

To reinstall after a machine move:

    cp ops/com.raahimnawaz.apex-*.plist ~/Library/LaunchAgents/
    launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.raahimnawaz.apex-refresh.plist
    launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.raahimnawaz.apex-qualifying.plist

---

## If you change the model

Run `make calibrate` and let the numbers decide. Every model choice in this repo was
settled that way, including two that went against my prior. Publish the result whatever
it says — the dashboard already renders a losing verdict correctly, and the significance
test is computed from the data rather than written into the copy.

Do not upgrade "promising" to "proven" until t clears the threshold at the actual
sample size.
