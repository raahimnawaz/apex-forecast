"""Layer 0 — pace deconvolution.

A raw lap time is mostly *not* car pace. It is fuel load, tyre age, track evolution,
traffic and flags, with the thing we actually want buried inside. This module fits one
robust regression per race and returns fuel/tyre/traffic-corrected pace per driver plus
per-compound degradation slopes.

Two identifiability constraints shape the model, and neither can be regressed away:

1. **Driver and team are collinear within a single race.** Every team fields exactly two
   cars, so a per-race fit cannot separate "the car is quick" from "the driver is quick" —
   the team effect is definitionally the mean of its two drivers. So Layer 0 estimates
   *driver* effects only, per race. Splitting those into constructor advantage vs driver
   skill needs variation across races and team-mate pairings, which is Layer 1's job.

2. **Fuel burn and track evolution are not separately identifiable from race laps alone.**
   Both are smooth monotone functions of lap number over a single stint-structured race.
   Rather than assume a textbook kg/lap coefficient and pretend the split is measured, the
   model carries one honest `lap_trend` term and reports it as the combined effect.

   A related condition binds the `lap_trend` term against degradation: within a stint,
   tyre age is an affine function of lap number, so if every driver pitted on the same
   lap the trend and the per-compound slopes would be perfectly collinear and only their
   sum would be recoverable. Real races identify them separately *because* drivers stop
   at different laps — the more uniform the field's strategy, the weaker that separation
   and the more the two terms trade off. `tests/test_pace.py` pins this down: the
   recovery tests only pass once the synthetic field runs varied pit windows.

Robust (Huber) regression is used instead of hard outlier cutoffs so that traffic laps and
lift-and-coast laps are downweighted continuously rather than by an arbitrary threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# A lap slower than this multiple of the event's green-flag median is a restart, a
# heavily compromised lap, or a data error. Huber handles moderate outliers; this only
# removes the ones that would distort the scale estimate itself.
OUTLIER_MULT = 1.12

# Dirty air saturates: past this gap the car ahead stops mattering aerodynamically.
DIRTY_AIR_HORIZON_S = 2.5

MIN_LAPS_PER_DRIVER = 8
MIN_LAPS_PER_EVENT = 120


@dataclass
class EventPace:
    """Deconvolved pace for one race."""
    season: int
    round: int
    event: str
    drivers: pd.DataFrame          # Driver, Team, pace_s, se_s, n_laps
    deg: pd.DataFrame              # compound, deg_s_per_lap, se, n_laps
    lap_trend_s: float             # combined fuel-burn + track-evolution, s/lap (negative = getting faster)
    dirty_air_s: float             # seconds lost at zero gap to the car ahead
    resid_sd_s: float
    n_laps: int
    baseline_s: float              # event green-flag median lap, for converting to %
    diagnostics: dict = field(default_factory=dict)


def add_gap_ahead(df: pd.DataFrame) -> pd.DataFrame:
    """Seconds to the car ahead on the same lap, from the session clock.

    Cars a lap down carry a different LapNumber and so drop out naturally rather than
    being compared against traffic they are not actually following.
    """
    df = df.sort_values(["LapNumber", "lap_end_s"]).copy()
    df["gap_ahead_s"] = df.groupby("LapNumber", sort=False)["lap_end_s"].diff()
    # Race leader has no car ahead: treat as clear air.
    df["gap_ahead_s"] = df["gap_ahead_s"].fillna(np.inf)
    df["dirty_air"] = np.clip(DIRTY_AIR_HORIZON_S - df["gap_ahead_s"], 0.0, DIRTY_AIR_HORIZON_S)
    return df


def prepare_race_laps(laps: pd.DataFrame, season: int, rnd: int) -> pd.DataFrame:
    """Green-flag, non-pit, representative race laps with model features attached."""
    d = laps[(laps["season"] == season) & (laps["round"] == rnd) & (laps["session"] == "R")].copy()
    if d.empty:
        return d

    d = d[d["lap_time_s"].notna()]
    d = d[d["IsAccurate"].astype(bool)]
    d = d[d["is_green"].astype(bool)]                 # excludes SC / VSC / yellow / red
    d = d[~d["is_pit_in"].astype(bool) & ~d["is_pit_out"].astype(bool)]
    d = d[d["LapNumber"] > 2]                         # start phase is its own process
    if d.empty:
        return d

    d = add_gap_ahead(d)

    med = d["lap_time_s"].median()
    d = d[d["lap_time_s"] < med * OUTLIER_MULT]

    d = d.rename(columns={"TyreLife": "tyre_age"})
    d["tyre_age"] = pd.to_numeric(d["tyre_age"], errors="coerce")
    d = d[d["tyre_age"].notna()]
    d["lap_c"] = d["LapNumber"] - d["LapNumber"].mean()
    d["Compound"] = d["Compound"].fillna("UNKNOWN").astype(str)

    # Drivers with too few surviving laps cannot support a stable effect.
    counts = d["Driver"].value_counts()
    d = d[d["Driver"].isin(counts[counts >= MIN_LAPS_PER_DRIVER].index)]

    # A compound needs enough laps to support a slope.
    ccounts = d["Compound"].value_counts()
    d = d[d["Compound"].isin(ccounts[ccounts >= 15].index)]

    return d


def fit_event(laps: pd.DataFrame, season: int, rnd: int) -> EventPace | None:
    """Fit the deconvolution model for one race."""
    d = prepare_race_laps(laps, season, rnd)
    if d.empty or len(d) < MIN_LAPS_PER_EVENT or d["Driver"].nunique() < 8:
        return None

    event = str(d["event"].iloc[0])
    baseline = float(d["lap_time_s"].median())

    formula = "lap_time_s ~ C(Driver) + C(Compound) + C(Compound):tyre_age + lap_c + dirty_air"
    try:
        res = smf.rlm(formula, data=d).fit()
    except Exception:  # noqa: BLE001
        return None

    params, bse = res.params, res.bse

    # --- driver effects, re-centred on the field mean (negative = faster) --------------
    levels = sorted(d["Driver"].unique())
    eff, se = {}, {}
    for drv in levels:
        key = f"C(Driver)[T.{drv}]"
        eff[drv] = float(params.get(key, 0.0))       # reference level has coefficient 0
        se[drv] = float(bse.get(key, np.nan))
    mean_eff = float(np.mean(list(eff.values())))

    team_of = d.groupby("Driver")["Team"].first()
    laps_of = d.groupby("Driver").size()
    drivers = pd.DataFrame({
        "Driver": levels,
        "Team": [team_of.get(x) for x in levels],
        "pace_s": [eff[x] - mean_eff for x in levels],
        "se_s": [se[x] for x in levels],
        "n_laps": [int(laps_of.get(x, 0)) for x in levels],
    })
    # The reference level of the treatment coding is pinned at 0 and so has no standard
    # error of its own. Every driver sits in the same design with a similar lap count, so
    # the field median SE is the right order of magnitude — imputed rather than left NaN,
    # and flagged so the substitution is never mistaken for an estimate.
    drivers["se_imputed"] = drivers["se_s"].isna()
    med_se = float(np.nanmedian(drivers["se_s"])) if drivers["se_s"].notna().any() else 0.0
    drivers["se_s"] = drivers["se_s"].fillna(med_se)

    drivers["pace_pct"] = 100.0 * drivers["pace_s"] / baseline
    drivers = drivers.sort_values("pace_s").reset_index(drop=True)

    # --- degradation slope per compound ------------------------------------------------
    deg_rows = []
    for comp in sorted(d["Compound"].unique()):
        key = f"C(Compound)[{comp}]:tyre_age"
        if key not in params:
            key = f"C(Compound)[T.{comp}]:tyre_age"
        if key in params:
            deg_rows.append({
                "compound": comp,
                "deg_s_per_lap": float(params[key]),
                "se": float(bse.get(key, np.nan)),
                "n_laps": int((d["Compound"] == comp).sum()),
            })
    deg = pd.DataFrame(deg_rows)

    resid = np.asarray(res.resid, dtype=float)
    resid_sd = float(np.nanstd(resid))
    ss_res = float(np.nansum(resid**2))
    ss_tot = float(np.nansum((d["lap_time_s"] - d["lap_time_s"].mean()) ** 2))

    return EventPace(
        season=season, round=rnd, event=event,
        drivers=drivers, deg=deg,
        lap_trend_s=float(params.get("lap_c", np.nan)),
        dirty_air_s=float(params.get("dirty_air", np.nan)) * DIRTY_AIR_HORIZON_S,
        resid_sd_s=resid_sd,
        n_laps=len(d),
        baseline_s=baseline,
        diagnostics={
            "pseudo_r2": round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else None,
            "n_drivers": int(d["Driver"].nunique()),
            "compounds": sorted(d["Compound"].unique().tolist()),
        },
    )


def fit_season(laps: pd.DataFrame, season: int) -> list[EventPace]:
    rounds = sorted(laps.loc[laps["season"] == season, "round"].unique())
    out = []
    for rnd in rounds:
        ep = fit_event(laps, season, int(rnd))
        if ep is not None:
            out.append(ep)
    return out


def season_pace_table(events: list[EventPace]) -> pd.DataFrame:
    """Long table of per-driver corrected pace across every fitted race."""
    frames = []
    for ep in events:
        f = ep.drivers.copy()
        f["season"], f["round"], f["event"] = ep.season, ep.round, ep.event
        f["baseline_s"], f["resid_sd_s"] = ep.baseline_s, ep.resid_sd_s
        frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
