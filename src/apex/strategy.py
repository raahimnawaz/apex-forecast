"""Layer 2a — pit strategy.

The gap this closes
-------------------
Layer 0 already recovers per-compound degradation slopes and Layer 1 already ranks cars,
but nothing connected the two: the forecast knew a compound lost 0.13 s/lap at Shanghai
and knew Aston Martin's tyres fall away faster than Mercedes', and did nothing with
either. This module turns those into a race time and then into a forecast offset.

The formulation is Bekker & Lotz (2009), *Planning Formula One race strategies using
discrete-event simulation* (J. Oper. Res. Soc. 60(7), 952-961), reduced to its analytic
core. They model a race as a sequence of stints whose cost is the integral of a tyre
degradation curve, punctuated by pit stops each costing a fixed pit-lane loss, and search
over the number and timing of stops for the plan that minimises total race time. The
lap-time decomposition — base pace + compound offset + degradation x tyre age, with pit
loss charged at each stop — is the one in Heilmeier, Graf & Lienkamp (2018), *A Race
Simulation for Strategy Decisions in Circuit Motorsports* (IEEE ITSC, 2986-2993), whose
Monte Carlo successor already underpins `reliability.py`.

So for a plan with stints of length n_1..n_k on compounds c_1..c_k:

    T = sum_j [ n_j * (base + offset_{c_j}) + deg_{c_j} * n_j (n_j - 1) / 2 ]
        + (k - 1) * pit_loss

The degradation term is the sum of an arithmetic series because degradation is linear in
tyre age, which is what Layer 0 fits. `optimise_plan` minimises T exactly over integer
stint lengths and compound assignments by dynamic programming — small enough to be exact,
so no search heuristic is involved and no local optimum can be blamed for a bad answer.

Why this is not double counting
-------------------------------
The obvious objection: Layer 1 fits *finishing orders*, which already reflect whatever
strategies teams actually ran. A team whose tyres fall apart already finishes lower, and
its strength term already absorbs that. Adding a raw strategy cost on top would count the
same weakness twice and inflate it.

What Layer 1 cannot know is that the cost is **circuit-specific**. Its strength term is a
season-average: it says Aston Martin is slow, not that Aston Martin's particular weakness
is punished at Barcelona (0.06 s/lap degradation, two stops) and nearly free at Monaco
(where track position outweighs tyre life and nobody can pass anyway). So the offset
applied here is the *deviation* of a car's strategy cost at this circuit from its own
season-average strategy cost:

    offset_i = -(S_i(circuit) - mean_over_training_circuits S_i) * seconds_to_theta

The season-average part is subtracted off precisely because Layer 1 already has it. What
is left is the part Layer 1 structurally cannot see. That also makes the layer
self-neutralising at an average circuit, which is the correct null behaviour.

The seconds-to-theta scale is fitted, not assumed — see `seconds_to_theta`.

Honest limits
-------------
- **Degradation is linear.** Real tyres fall off a cliff. Layer 0 fits a slope because a
  slope is what ~25 laps of stint data supports; the optimiser inherits that assumption
  and will therefore understate the value of an extra stop on a compound that cliffs.
- **Stops are assumed green.** Pit loss is measured only from stops where both in- and
  out-lap ran green, because a safety-car stop is roughly half price. The safety-car
  process in `reliability.py` raises the entropy of the order rather than handing anyone a
  cheap stop, so the interaction between the two layers is not modelled.
- **No undercut, no traffic, no tyre warm-up.** This is a free-air time optimum, not a
  race. Two cars whose optimal plans differ by two seconds will not necessarily finish in
  that order.
- **Per-team degradation is weak per race** (offsets ~0.02-0.04 s/lap against standard
  errors ~0.011). It is only usable pooled across the season, which is what
  `fit_team_degradation` does, and it is shrunk toward zero by how much of the spread
  between teams is real rather than noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .pace import prepare_race_laps

# Stops beyond this are not a real F1 strategy; the search stops there rather than
# discovering a 5-stop optimum that no team would ever run.
MAX_STOPS = 3

# A stint shorter than this is a tyre change, not a stint. Prevents the optimiser from
# proposing a 1-lap stint to dodge the degradation integral.
MIN_STINT = 5

# The dry-race rule: at least two different compounds must be used. This is a sporting
# regulation, not a modelling choice, and it is what makes a zero-stop plan illegal.
MIN_COMPOUNDS = 2

# Pooling strength for per-circuit pit loss, in pseudo-observations at the season mean.
# Barcelona has 40 measured stops and needs almost none; the Chinese GP has 4 and needs
# a lot. Same device as the per-circuit safety-car rates in reliability.py.
PIT_LOSS_PRIOR_K = 8.0

# Compounds the optimiser is allowed to choose between. Wets are excluded: a wet race is
# a different process and this project has one of them.
DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


@dataclass
class TyreModel:
    """Per-circuit tyre behaviour and per-team degradation, both estimated from laps."""
    by_circuit: pd.DataFrame     # event, round, compound, deg_s_per_lap, offset_s, n_laps
    by_team: pd.DataFrame        # team, deg_offset_s, se, shrunk, n_races
    pit_loss: pd.DataFrame       # event, round, stops, pit_loss_s, pit_loss_raw
    race_laps: pd.DataFrame      # event, round, n_laps
    shrinkage: float             # fraction of the raw team spread kept as signal


@dataclass
class Plan:
    """One strategy: which compounds, how long on each, and what it costs."""
    stints: tuple[int, ...]
    compounds: tuple[str, ...]
    total_s: float               # race time above the base pace: deg + offsets + pit loss
    n_stops: int


# --------------------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------------------

def measure_pit_loss(laps: pd.DataFrame, season: int) -> pd.DataFrame:
    """Seconds lost to a green-flag pit stop, per circuit.

    Measured as the excess of the in-lap and out-lap over that driver's own median green
    lap, summed. Using the driver's own median rather than the field's keeps a slow car's
    slow laps from inflating its apparent pit loss.

    Stops where either lap ran under a safety car, VSC or yellow are dropped. That is not
    fastidiousness: a stop under a safety car costs roughly half as much because the whole
    field is slowed, so mixing the two would bias every circuit's estimate downward by an
    amount that depends on how many safety cars it happened to have.
    """
    r = laps[(laps["season"] == season) & (laps["session"] == "R")].copy()
    rows = []
    for (rnd, ev), g in r.groupby(["round", "event"], sort=True):
        for _, gd in g.groupby("Driver", sort=False):
            gd = gd.sort_values("LapNumber")
            clean = gd[gd["is_green"].astype(bool) & gd["IsAccurate"].astype(bool)
                       & ~gd["is_pit_in"].astype(bool) & ~gd["is_pit_out"].astype(bool)]
            med = clean["lap_time_s"].median()
            if not np.isfinite(med):
                continue
            by_lap = gd.set_index("LapNumber")
            for lap in gd.loc[gd["is_pit_in"].astype(bool), "LapNumber"]:
                if lap not in by_lap.index or (lap + 1) not in by_lap.index:
                    continue
                pair = by_lap.loc[[lap, lap + 1]]
                if not pair["is_green"].astype(bool).all():
                    continue
                t = pair["lap_time_s"].to_numpy(dtype=float)
                if not np.isfinite(t).all():
                    continue
                rows.append({"round": int(rnd), "event": ev, "loss": float(t.sum() - 2 * med)})

    if not rows:
        return pd.DataFrame(columns=["event", "round", "stops", "pit_loss_s", "pit_loss_raw"])

    p = pd.DataFrame(rows)
    season_med = float(p["loss"].median())
    out = (p.groupby(["round", "event"], as_index=False)["loss"]
             .agg(stops="size", pit_loss_raw="median"))
    # Shrink toward the season median by measured-stop count. A circuit with four stops
    # has no business asserting a pit loss 10 s off the norm.
    w = out["stops"] / (out["stops"] + PIT_LOSS_PRIOR_K)
    out["pit_loss_s"] = w * out["pit_loss_raw"] + (1 - w) * season_med
    return out[["event", "round", "stops", "pit_loss_s", "pit_loss_raw"]]


def _fit_race_tyres(laps: pd.DataFrame, season: int, rnd: int) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Per-compound degradation and offset, plus per-team degradation offsets, for one race.

    This is Layer 0's model with one term added: `C(Team):tyre_age`, a per-team modifier on
    the degradation slope. Layer 0's own production fit is deliberately left alone — it is
    validated against Layer 1 at rho = 0.96 and is not worth perturbing to serve a
    downstream consumer — so the extra term is fitted here instead.
    """
    d = prepare_race_laps(laps, season, rnd)
    if d.empty or d["Team"].nunique() < 4:
        return None
    formula = ("lap_time_s ~ C(Driver) + C(Compound) + C(Compound):tyre_age"
               " + C(Team):tyre_age + lap_c + dirty_air")
    try:
        res = smf.rlm(formula, data=d).fit()
    except Exception:  # noqa: BLE001 - a race that will not fit simply contributes nothing
        return None

    params, bse = res.params, res.bse
    event = str(d["event"].iloc[0])

    comp_rows = []
    for comp in sorted(d["Compound"].unique()):
        if comp not in DRY_COMPOUNDS:
            continue
        slope = params.get(f"C(Compound)[{comp}]:tyre_age",
                           params.get(f"C(Compound)[T.{comp}]:tyre_age"))
        off = params.get(f"C(Compound)[{comp}]", params.get(f"C(Compound)[T.{comp}]", 0.0))
        if slope is None:
            continue
        comp_rows.append({
            "season": season, "round": rnd, "event": event, "compound": comp,
            "deg_s_per_lap": float(slope), "offset_s": float(off),
            "n_laps": int((d["Compound"] == comp).sum()),
        })

    team_rows = []
    for key in params.index:
        if key.startswith("C(Team)") and key.endswith(":tyre_age"):
            team = key.split("[T.", 1)[-1].split("]", 1)[0]
            team_rows.append({"round": rnd, "team": team, "deg_offset_s": float(params[key]),
                              "se": float(bse.get(key, np.nan))})

    # Treatment coding pins one team at zero, so the offsets are relative to whichever
    # team sorted first. Re-centre on the field so they read as "faster or slower
    # degrading than the average car", which is what the optimiser wants.
    t = pd.DataFrame(team_rows)
    if not t.empty:
        present = sorted(d["Team"].unique())
        missing = [x for x in present if x not in set(t["team"])]
        for m in missing:                                   # the reference level
            t = pd.concat([t, pd.DataFrame([{"round": rnd, "team": m, "deg_offset_s": 0.0,
                                             "se": float(np.nanmedian(t["se"]))}])],
                          ignore_index=True)
        t["deg_offset_s"] -= t["deg_offset_s"].mean()

    return pd.DataFrame(comp_rows), t


def fit_team_degradation(laps: pd.DataFrame, season: int) -> TyreModel:
    """Estimate per-circuit tyre behaviour and per-team degradation across a season.

    Per race, a team's degradation offset is estimated to about +/-0.011 s/lap against a
    between-team spread of 0.02-0.04 s/lap, so a single race cannot support a per-team
    number. Pooling is what makes it usable: offsets are combined across races by inverse
    variance, then shrunk toward zero by the share of the observed between-team spread
    that survives subtracting the measurement noise —

        shrinkage = max(0, 1 - mean(se^2) / var(offsets))

    which is the standard empirical-Bayes estimate of how much of an observed spread is
    real. If the teams differ no more than their own error bars, this is zero and the
    layer switches itself off rather than forecasting from noise.
    """
    rounds = sorted(laps.loc[laps["season"] == season, "round"].unique())
    comp_frames, team_frames = [], []
    for rnd in rounds:
        got = _fit_race_tyres(laps, season, int(rnd))
        if got is None:
            continue
        c, t = got
        if not c.empty:
            comp_frames.append(c)
        if not t.empty:
            team_frames.append(t)

    by_circuit = (pd.concat(comp_frames, ignore_index=True) if comp_frames
                  else pd.DataFrame(columns=["season", "round", "event", "compound",
                                             "deg_s_per_lap", "offset_s", "n_laps"]))

    if team_frames:
        tt = pd.concat(team_frames, ignore_index=True)
        tt["se"] = tt["se"].fillna(tt["se"].median())
        tt["w"] = 1.0 / np.maximum(tt["se"] ** 2, 1e-9)
        agg = tt.groupby("team").apply(
            lambda g: pd.Series({
                "deg_offset_s": float(np.average(g["deg_offset_s"], weights=g["w"])),
                "se": float(np.sqrt(1.0 / g["w"].sum())),
                "n_races": len(g),
            }), include_groups=False).reset_index()
        agg["deg_offset_s"] -= agg["deg_offset_s"].mean()

        var_obs = float(np.var(agg["deg_offset_s"], ddof=1)) if len(agg) > 1 else 0.0
        noise = float(np.mean(agg["se"] ** 2))
        shrink = max(0.0, 1.0 - noise / var_obs) if var_obs > 0 else 0.0
        agg["shrunk"] = agg["deg_offset_s"] * shrink
        by_team = agg.sort_values("shrunk").reset_index(drop=True)
    else:
        by_team = pd.DataFrame(columns=["team", "deg_offset_s", "se", "n_races", "shrunk"])
        shrink = 0.0

    race = laps[(laps["season"] == season) & (laps["session"] == "R")]
    race_laps = (race.groupby(["round", "event"], as_index=False)["LapNumber"].max()
                     .rename(columns={"LapNumber": "n_laps"}))
    race_laps["n_laps"] = race_laps["n_laps"].astype(int)

    # The longest stint anyone actually ran here, which bounds the optimiser. Degradation
    # is fitted as a straight line over the stint lengths teams chose; asked for the cost
    # of a stint longer than any in the data, a straight line answers confidently and
    # wrongly, because real tyres fall off a cliff that a slope cannot represent. Capping
    # at the observed maximum keeps the optimiser inside the range the model was fitted on.
    dry = race[race["Compound"].isin(DRY_COMPOUNDS)]
    if not dry.empty:
        stints = (dry.groupby(["round", "event", "Driver", "Stint"], as_index=False)
                     .agg(n=("LapNumber", "size")))
        longest = (stints.groupby(["round", "event"], as_index=False)["n"].max()
                         .rename(columns={"n": "max_stint"}))
        race_laps = race_laps.merge(longest, on=["round", "event"], how="left")
    if "max_stint" not in race_laps.columns:
        race_laps["max_stint"] = race_laps["n_laps"]
    race_laps["max_stint"] = (race_laps["max_stint"].fillna(race_laps["n_laps"])
                              .astype(int))

    return TyreModel(by_circuit=by_circuit, by_team=by_team,
                     pit_loss=measure_pit_loss(laps, season),
                     race_laps=race_laps, shrinkage=float(shrink))


def season_average_profile(tyres: TyreModel, event: str, rnd: int) -> pd.DataFrame:
    """A tyre profile for a circuit whose race has not happened, averaged over those that have.

    A forecast cannot use the target race's own degradation — that is the race it is trying
    to predict. So an unraced circuit gets the mean per-compound slope and offset across
    every race fitted so far. What stays genuinely circuit-specific is the race distance
    and the pit loss, both of which are known or estimable before the cars run, and both of
    which move the optimum: 78 laps at Monaco and 44 at Spa are different strategy problems
    even under an identical tyre model.

    **Why not fit the profile from Friday practice?** It was tried, and it does not work.
    Practice long runs give the wrong sign: fitted degradation came out at -0.29 s/lap at
    Spa and -0.07 at the Hungaroring against race values of +0.06 and +0.08, and +3.86 at
    a wet Canadian weekend. The reason is structural rather than a matter of filtering.
    Within a single long run the car burns fuel as the tyre ages, so fuel load and tyre age
    are both linear in lap-within-stint and therefore perfectly collinear — the same
    identifiability wall `pace.py` documents, except that a race escapes it because drivers
    pit at different laps and practice has no such shared clock. Separating them needs an
    assumed kg/lap fuel coefficient, and inventing that number to make a layer work is the
    thing this project refuses to do elsewhere. The practice laps stay unused here for a
    stated reason rather than an oversight.
    """
    if tyres.by_circuit.empty:
        return pd.DataFrame(columns=["season", "round", "event", "compound",
                                     "deg_s_per_lap", "offset_s", "n_laps"])
    src = tyres.by_circuit[tyres.by_circuit["event"] != event]
    if src.empty:
        src = tyres.by_circuit
    agg = (src.groupby("compound", as_index=False)
              .agg(deg_s_per_lap=("deg_s_per_lap", "mean"),
                   offset_s=("offset_s", "mean"), n_laps=("n_laps", "sum")))
    agg["season"] = int(src["season"].iloc[0])
    agg["round"] = rnd
    agg["event"] = event
    return agg[["season", "round", "event", "compound", "deg_s_per_lap", "offset_s", "n_laps"]]


def with_circuit_profile(tyres: TyreModel, profile: pd.DataFrame,
                         n_laps: int | None = None,
                         pit_loss: float | None = None) -> TyreModel:
    """Return a copy of `tyres` with one circuit's profile replaced or added.

    Used to hand the optimiser a practice-derived profile for a circuit whose race has not
    happened, without disturbing the team degradation estimates that were pooled across
    every race so far. Race length and pit loss fall back to season medians when the
    circuit is genuinely new, since neither can be measured before the event.
    """
    if profile.empty:
        return tyres
    event = str(profile["event"].iloc[0])
    rnd = int(profile["round"].iloc[0])

    by_circuit = pd.concat(
        [tyres.by_circuit[tyres.by_circuit["event"] != event], profile], ignore_index=True)

    rl = tyres.race_laps[tyres.race_laps["event"] != event]
    prior = tyres.race_laps[tyres.race_laps["event"] == event]
    n = (int(n_laps) if n_laps is not None
         else int(prior["n_laps"].iloc[0]) if not prior.empty
         else int(tyres.race_laps["n_laps"].median()) if not tyres.race_laps.empty else 0)
    ms = (int(prior["max_stint"].iloc[0]) if not prior.empty
          else int(tyres.race_laps["max_stint"].median()) if not tyres.race_laps.empty else n)
    rl = pd.concat([rl, pd.DataFrame([{"round": rnd, "event": event, "n_laps": n,
                                       "max_stint": ms}])], ignore_index=True)

    pl = tyres.pit_loss[tyres.pit_loss["event"] != event]
    prior_pl = tyres.pit_loss[tyres.pit_loss["event"] == event]
    loss = (float(pit_loss) if pit_loss is not None
            else float(prior_pl["pit_loss_s"].iloc[0]) if not prior_pl.empty
            else float(tyres.pit_loss["pit_loss_s"].median()) if not tyres.pit_loss.empty
            else 22.0)
    pl = pd.concat([pl, pd.DataFrame([{"event": event, "round": rnd, "stops": 0,
                                       "pit_loss_s": loss, "pit_loss_raw": np.nan}])],
                   ignore_index=True)

    return TyreModel(by_circuit=by_circuit, by_team=tyres.by_team, pit_loss=pl,
                     race_laps=rl, shrinkage=tyres.shrinkage)


# --------------------------------------------------------------------------------------
# the optimiser
# --------------------------------------------------------------------------------------

def _stint_cost(n: int, deg: float, offset: float) -> float:
    """Cost of one stint above base pace: compound offset plus the degradation integral.

    Degradation is linear in tyre age, so a stint of n laps starting on a fresh set costs
    deg * (0 + 1 + ... + n-1) = deg * n(n-1)/2 above what the same laps would cost on a
    tyre that never aged.
    """
    return n * offset + deg * n * (n - 1) / 2.0


def optimise_plan(n_laps: int, pit_loss: float, deg: dict[str, float],
                  offset: dict[str, float], max_stops: int = MAX_STOPS,
                  min_stint: int = MIN_STINT,
                  min_compounds: int = MIN_COMPOUNDS,
                  max_stint: int | None = None) -> Plan | None:
    """Exact minimum-time strategy by dynamic programming over integer stint lengths.

    State is (stint index, laps completed, set of compounds used so far). With at most
    four stints, ~78 laps and three compounds that is a few thousand states, so the
    optimum is found exactly rather than searched for. Bekker & Lotz reached the same
    answer by discrete-event simulation; with linear degradation the objective is
    separable across stints and a DP does it in closed form.

    Returns None when no legal plan exists — too few compounds, or a race too short to
    fit the mandatory two.
    """
    comps = [c for c in DRY_COMPOUNDS if c in deg and c in offset]
    if not comps or n_laps < min_stint * 2:
        return None
    ix = {c: i for i, c in enumerate(comps)}
    cap = n_laps if max_stint is None else max(int(max_stint), min_stint)

    best: Plan | None = None
    for n_stints in range(2, max_stops + 2):
        if n_stints * min_stint > n_laps:
            break

        # layers[s][(laps_used, compounds_mask)] = (cost, parent_key, stint_len, compound).
        # Every layer is kept so the winning plan can be walked back out; the state space
        # is a few thousand entries, so holding all of it costs nothing.
        layers: list[dict[tuple[int, int], tuple[float, tuple | None, int, str]]] = [
            {(0, 0): (0.0, None, 0, "")}
        ]
        for s in range(n_stints):
            nxt: dict[tuple[int, int], tuple[float, tuple | None, int, str]] = {}
            remaining = n_stints - s - 1
            for key, (cost, _, _, _) in layers[-1].items():
                used, mask = key
                # Leave enough laps for the stints still to come, never overshoot, and
                # never propose a stint longer than anyone ran here.
                hi = min(n_laps - used - remaining * min_stint, cap)
                for n in range(min_stint, hi + 1):
                    for c in comps:
                        nk = (used + n, mask | (1 << ix[c]))
                        nc = cost + _stint_cost(n, deg[c], offset[c])
                        if nk not in nxt or nc < nxt[nk][0]:
                            nxt[nk] = (nc, key, n, c)
            if not nxt:
                break
            layers.append(nxt)

        if len(layers) != n_stints + 1:
            continue

        final = layers[-1]
        for key, (cost, _, _, _) in final.items():
            used, mask = key
            if used != n_laps or mask.bit_count() < min_compounds:
                continue
            total = cost + (n_stints - 1) * pit_loss
            if best is not None and total >= best.total_s:
                continue
            stints, compounds, k, s = [], [], key, n_stints
            while s > 0:
                _, parent, n, c = layers[s][k]
                stints.append(n)
                compounds.append(c)
                k, s = parent, s - 1
            best = Plan(stints=tuple(reversed(stints)),
                        compounds=tuple(reversed(compounds)),
                        total_s=total, n_stops=n_stints - 1)
    return best


def _compound_maps(tyres: TyreModel, event: str,
                   team_offset: float = 0.0) -> tuple[dict[str, float], dict[str, float]] | None:
    """This circuit's degradation and compound offsets, with a team's degradation applied.

    The team modifier shifts every compound's slope by the same amount, because that is
    what Layer 0 estimates — one degradation offset per team, not one per team per
    compound. Slopes are floored at a small positive value: a negative fitted slope is a
    small-sample artefact, and left alone it would let the optimiser run one stint for the
    whole race and claim the tyre got faster.
    """
    c = tyres.by_circuit[tyres.by_circuit["event"] == event]
    if c.empty:
        return None
    deg, off = {}, {}
    for r in c.itertuples():
        deg[r.compound] = max(float(r.deg_s_per_lap) + team_offset, 1e-4)
        off[r.compound] = float(r.offset_s)
    # Offsets are only meaningful relative to each other; centring keeps the absolute
    # level out of the comparison between drivers.
    if off:
        m = float(np.mean(list(off.values())))
        off = {k: v - m for k, v in off.items()}
    return deg, off


def strategy_cost(tyres: TyreModel, event: str, teams: list[str],
                  n_laps: int | None = None,
                  pit_loss: float | None = None) -> tuple[np.ndarray, list[Plan | None]]:
    """Optimal-strategy race time above base pace, per team, at one circuit.

    Returns (cost_s, plans). Cost is what the tyres and the pit lane take out of a car
    over the full race distance when it is driven to its own best plan — so a team that
    degrades badly pays for it twice, once in the degradation integral and again in the
    extra stop it is forced into.
    """
    row = tyres.race_laps[tyres.race_laps["event"] == event]
    max_stint = int(row["max_stint"].iloc[0]) if not row.empty else None
    if n_laps is None:
        n_laps = int(row["n_laps"].iloc[0]) if not row.empty else 0
    if pit_loss is None:
        row = tyres.pit_loss[tyres.pit_loss["event"] == event]
        pit_loss = (float(row["pit_loss_s"].iloc[0]) if not row.empty
                    else float(tyres.pit_loss["pit_loss_s"].median())
                    if not tyres.pit_loss.empty else 22.0)

    tmap = tyres.by_team.set_index("team")["shrunk"].to_dict() if not tyres.by_team.empty else {}
    costs, plans = [], []
    for t in teams:
        maps = _compound_maps(tyres, event, float(tmap.get(t, 0.0)))
        if maps is None or not n_laps:
            costs.append(np.nan)
            plans.append(None)
            continue
        deg, off = maps
        p = optimise_plan(n_laps, pit_loss, deg, off, max_stint=max_stint)
        costs.append(np.nan if p is None else p.total_s)
        plans.append(p)
    return np.asarray(costs, dtype=float), plans


def seconds_to_theta(pace: pd.DataFrame, strength: pd.DataFrame) -> float:
    """Fit how many theta units one second per lap is worth.

    Layer 1's strength is on a log-odds scale with no intrinsic units, so a race-time
    delta cannot be added to it until the exchange rate is measured. Layer 0 gives
    per-team corrected pace in seconds per lap and Layer 1 gives per-team strength; the
    two are already known to agree at rho = 0.96, and the slope of that relationship is
    the conversion.

    Regressing strength on pace (not the reverse) is the right direction here: pace is the
    quantity being converted *from*, so its coefficient is the factor wanted. Returns 0.0
    when the fit is not supportable, which switches the strategy layer off rather than
    letting it apply an arbitrary scale.
    """
    if pace.empty or strength.empty:
        return 0.0
    p = pace.groupby("Team")["pace_s"].mean()
    m = strength.set_index("constructor").join(p).dropna(subset=["pace_s", "car_2026_latest"])
    if len(m) < 4:
        return 0.0
    x = m["pace_s"].to_numpy(dtype=float)
    y = m["car_2026_latest"].to_numpy(dtype=float)
    if np.var(x) <= 0:
        return 0.0
    slope = float(np.polyfit(x, y, 1)[0])
    # Faster pace is a *lower* pace_s, so the slope is negative and the magnitude is the
    # exchange rate. A positive slope would mean slower cars rank higher, which is a
    # broken fit, not a finding — refuse it rather than propagate it.
    return abs(slope) if slope < 0 else 0.0


def theta_offsets(tyres: TyreModel, event: str, teams: list[str], scale: float,
                  training_events: list[str] | None = None) -> np.ndarray:
    """Circuit-specific strategy offsets in theta units, one per entry.

    The offset is the deviation of a team's strategy cost at *this* circuit from its own
    average across the training circuits, converted to theta. Subtracting the team's own
    mean is what keeps this from double counting Layer 1: the season-average penalty for
    degrading badly is already in the strength term, and only the circuit-specific part is
    new information. A team with average tyre behaviour, or a circuit that punishes tyres
    averagely, gets an offset of zero.

    The result is then centred across the field, for the same reason `grid_adv` is centred
    per race in `strength.py`: a constant added to every entry is invisible to a
    Plackett-Luce, so the part of a circuit's strategy cost that everyone pays is not a
    forecast signal at all. Uncentred it also dwarfs the part that is — at the Hungaroring
    the shared component is about -2.0 theta against a between-team spread of 0.1 — which
    would make the layer look powerful while doing almost nothing. Centring leaves exactly
    the re-ranking signal and nothing else.

    Sign: a *higher* strategy cost is worse, so it enters theta negatively.
    """
    if scale <= 0 or tyres.by_circuit.empty:
        return np.zeros(len(teams))

    here, _ = strategy_cost(tyres, event, teams)
    if training_events is None:
        training_events = [e for e in tyres.race_laps["event"].unique() if e != event]
    if not training_events:
        return np.zeros(len(teams))

    ref = np.vstack([strategy_cost(tyres, e, teams)[0] for e in training_events])
    baseline = np.nanmean(ref, axis=0)

    n_laps_here = tyres.race_laps.loc[tyres.race_laps["event"] == event, "n_laps"]
    n = int(n_laps_here.iloc[0]) if not n_laps_here.empty else 0
    if not n:
        return np.zeros(len(teams))

    # Race-time deltas are converted to a per-lap rate before applying the exchange rate,
    # because the scale was fitted against pace in seconds per lap.
    delta = np.where(np.isfinite(here) & np.isfinite(baseline), here - baseline, 0.0)
    out = -(delta / n) * scale
    return out - out.mean()

