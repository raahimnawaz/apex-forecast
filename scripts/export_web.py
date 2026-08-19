"""Serialise the model outputs into the versioned JSON the static dashboard reads.

The dashboard never touches a database or an upstream API at view time — it loads
these files and nothing else. That keeps it deployable to static hosting and keeps
Jolpica/FastF1 out of the request path.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from datetime import UTC, datetime

import fastf1
import fastf1.plotting
import numpy as np
import pandas as pd

from apex.paths import CACHE, PROCESSED, RAW, REPORTS, WEB_DATA

fastf1.Cache.enable_cache(str(CACHE))

# Official-ish team identity colours. Used ONLY as identity chips beside driver names,
# never as a chart's categorical scale — the real F1 palette has several near-identical
# blues and reds and fails contrast/CVD checks outright.
TEAM_FALLBACK = {
    "Mercedes": "#00d7b6", "Ferrari": "#e8002d", "Red Bull Racing": "#3671c6",
    "McLaren": "#ff8000", "Aston Martin": "#229971", "Alpine": "#0093cc",
    "Williams": "#64c4ff", "Racing Bulls": "#6692ff", "Audi": "#00e701",
    "Haas F1 Team": "#b6babd", "Cadillac": "#c8a55a",
}


def team_colors(season: int, teams: list[str]) -> dict[str, str]:
    out = {}
    for t in teams:
        c = TEAM_FALLBACK.get(t)
        if c is None:
            try:
                c = fastf1.plotting.get_team_color(t, session=None)
            except Exception:  # noqa: BLE001
                c = "#898781"
        out[t] = c
    return out


def clean(records: list[dict]) -> list[dict]:
    """NaN/Inf are not valid JSON. Emit null so the client can tell 'unknown' from zero."""
    out = []
    for rec in records:
        out.append({k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                    for k, v in rec.items()})
    return out


def next_event(season: int) -> dict:
    sched = fastf1.get_event_schedule(season, include_testing=False)
    now = pd.Timestamp.now().normalize()
    upcoming = sched[sched["EventDate"] >= now]
    row = upcoming.iloc[0] if not upcoming.empty else sched.iloc[-1]
    return {
        "round": int(row["RoundNumber"]),
        "name": str(row["EventName"]),
        "location": str(row.get("Location", "")),
        "country": str(row.get("Country", "")),
        "date": pd.Timestamp(row["EventDate"]).strftime("%Y-%m-%d"),
        "format": str(row["EventFormat"]),
        "total_rounds": int(sched["RoundNumber"].max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    args = ap.parse_args()
    season = args.season

    pace = pd.read_parquet(PROCESSED / f"pace_{season}.parquet")
    deg = pd.read_parquet(PROCESSED / f"degradation_{season}.parquet")

    # --- season-level corrected pace, with spread across races -------------------------
    agg = (pace.groupby(["Driver", "Team"], as_index=False)
                .agg(pace_s=("pace_s", "mean"),
                     sd_s=("pace_s", "std"),
                     races=("round", "nunique"),
                     laps=("n_laps", "sum")))
    agg["sd_s"] = agg["sd_s"].fillna(0.0)
    # Standard error of the driver's mean pace across races — the honest uncertainty band.
    agg["se_s"] = agg["sd_s"] / agg["races"].clip(lower=1) ** 0.5
    agg = agg.sort_values("pace_s").reset_index(drop=True)
    agg["rank"] = agg.index + 1

    teams = sorted(agg["Team"].dropna().unique().tolist())
    colors = team_colors(season, teams)

    # --- per-round pace, for the spread / form view -------------------------------------
    by_round = (pace[["Driver", "Team", "round", "event", "pace_s", "se_s", "n_laps"]]
                .sort_values(["round", "pace_s"]))

    # --- degradation ---------------------------------------------------------------------
    deg_agg = (deg.groupby("compound", as_index=False)
                  .agg(deg_s_per_lap=("deg_s_per_lap", "mean"),
                       sd=("deg_s_per_lap", "std"),
                       races=("round", "nunique"),
                       laps=("n_laps", "sum")))
    deg_agg["sd"] = deg_agg["sd"].fillna(0.0)
    order = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}
    deg_agg["order"] = deg_agg["compound"].map(order).fillna(9)
    deg_agg = deg_agg.sort_values("order").drop(columns="order")

    fit = pd.read_csv(REPORTS / f"pace_fit_{season}.csv")

    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season": season,
        "rounds_analysed": sorted(int(r) for r in pace["round"].unique()),
        "next_event": next_event(season),
        "team_colors": colors,
        "pace": clean(agg.round(4).to_dict(orient="records")),
        "pace_by_round": clean(by_round.round(4).to_dict(orient="records")),
        "degradation": clean(deg_agg.round(4).to_dict(orient="records")),
        "totals": {
            "laps_modelled": int(fit["n_laps"].sum()),
            "mean_dirty_air_cost_s": float(fit["dirty_air_cost_s"].mean()),
        },
    }

    out = WEB_DATA / f"pace_{season}.json"
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    print(f"  drivers: {len(agg)}  rounds: {payload['rounds_analysed']}")
    print(f"  laps modelled: {payload['totals']['laps_modelled']:,}")

    export_strength(season, colors)
    export_season(season)
    export_teams(season)
    return 0


def _need(path):
    if not path.exists():
        raise SystemExit(f"missing {path.name} — run scripts/build_strength.py first")
    return path


def calibration_payload() -> dict | None:
    """Walk-forward results, published whatever they say.

    A forecast that has not been scored out of sample is an opinion. These numbers are
    shipped to the page unchanged, including when the naive baseline wins — that is the
    entire point of running the test.
    """
    summary = REPORTS / "calibration_summary.csv"
    if not summary.exists():
        return None
    s = pd.read_csv(summary)
    rows = clean(s.round(4).to_dict(orient="records"))
    best = min(rows, key=lambda r: r["rps"])["model"]
    grid_rps = next((r["rps"] for r in rows if r["model"] == "baseline: grid"), None)

    # Winning on the mean is not the same as winning. With six races the margin has to be
    # tested, not just reported — otherwise the page overclaims on a difference that a
    # paired t-test cannot separate from noise.
    per_race = REPORTS / "calibration_walkforward.csv"
    sig = {}
    if per_race.exists() and grid_rps is not None:
        piv = pd.read_csv(per_race).pivot_table(index="round", columns="model", values="rps")
        if best in piv and "baseline: grid" in piv:
            d = piv["baseline: grid"] - piv[best]
            n = len(d)
            t = float(d.mean() / (d.std(ddof=1) / n ** 0.5)) if d.std(ddof=1) > 0 else 0.0
            # Critical value has to follow the sample size, not be pinned to whatever
            # the sample happened to be the first time this ran.
            from scipy.stats import t as tdist
            crit = float(tdist.ppf(0.975, df=max(n - 1, 1)))

            # Which metrics the model actually wins, rather than a blanket claim. On the
            # current sample the baseline still takes podium log-loss and Spearman.
            lower_better = {"rps": True, "ll_win": True, "ll_podium": True,
                            "ll_points": True, "spearman": False}
            bl = next((r for r in rows if r["model"] == "baseline: grid"), None)
            md = next((r for r in rows if r["model"] == best), None)
            beaten = ([k for k, lo in lower_better.items()
                       if md and bl and ((md[k] < bl[k]) if lo else (md[k] > bl[k]))]
                      if md and bl else [])
            sig = {
                "margin_mean": round(float(d.mean()), 5),
                "margin_sd": round(float(d.std(ddof=1)), 5),
                "races_won": int((d > 0).sum()),
                "t_stat": round(t, 2),
                "t_critical": round(crit, 2),
                "significant": bool(abs(t) > crit),
                "metrics_won": beaten,
                "metrics_total": len(lower_better),
            }

    return {
        "summary": rows,
        "best_model": best,
        "grid_baseline_rps": grid_rps,
        "beats_grid_baseline": bool(best.startswith("model:")),
        "n_eval_races": int(s["races"].max()),
        "significance": sig,
    }


def export_strength(season: int, colors: dict) -> None:
    """Layer 1 posterior + the next-race forecast."""
    skill = pd.read_parquet(_need(PROCESSED / f"skill_{season}.parquet"))
    cons = pd.read_parquet(_need(PROCESSED / f"constructor_{season}.parquet"))
    by_round = pd.read_parquet(PROCESSED / "constructor_by_round.parquet")
    fc = pd.read_parquet(PROCESSED / "forecast_next.parquet")
    entries = pd.read_parquet(PROCESSED / "forecast_entries.parquet")
    probs = np.load(PROCESSED / "position_probs.npy")
    diag = pd.read_csv(REPORTS / "strength_fit.csv").iloc[0].to_dict()

    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season": season,
        "team_colors": colors,
        "skill": clean(skill.round(4).to_dict(orient="records")),
        "constructor": clean(cons.round(4).to_dict(orient="records")),
        "constructor_by_round": clean(by_round.round(4).to_dict(orient="records")),
        "forecast": clean(fc.round(5).to_dict(orient="records")),
        "position_matrix": {
            "drivers": entries["driver"].tolist(),
            "teams": entries["team"].tolist(),
            "probs": np.round(probs, 5).tolist(),
        },
        "diagnostics": {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                        for k, v in diag.items()},
        "reliability": clean(pd.read_parquet(PROCESSED / "reliability.parquet")
                             .round(4).to_dict(orient="records"))
                       if (PROCESSED / "reliability.parquet").exists() else [],
        "weather": json.loads((PROCESSED / "weather_next.json").read_text())
                   if (PROCESSED / "weather_next.json").exists() else None,
        "calibration": calibration_payload(),
    }
    out = WEB_DATA / f"strength_{season}.json"
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    print(f"  constructor share {100 * diag['constructor_share']:.1f}% · "
          f"R-hat {diag['worst_rhat']} · Layer-0 rho {diag['layer0_spearman']}")
    _append_prediction_log(season, payload, diag)


def _append_prediction_log(season: int, payload: dict, diag: dict) -> None:
    """Write this forecast to an immutable per-round file, if one is not already there.

    `strength_{season}.json` always holds the *next* race, so it is overwritten on every
    build. That makes it useless as a record: scoring round 11 after the round 12 build has
    run silently grades the wrong forecast and reports plausible numbers for it. This is the
    prediction log `PLAN.md` calls the credibility feature — one file per round, written
    once and never rewritten, so a published forecast can still be checked months later.

    **Never overwritten by design.** A forecast that could be revised after the race is not
    evidence of anything, so a second build for the same round leaves the first file alone
    and says so. Delete the file by hand if a genuine re-publish is intended.
    """
    rnd = diag.get("forecast_round")
    if rnd is None or (isinstance(rnd, float) and not math.isfinite(rnd)):
        print("  no forecast round recorded — prediction log not written")
        return

    # A grid-free forecast is provisional. `status.py` calls it "correct until qualifying
    # runs", and limitation #1 is that the real forecast conditions on the grid and so
    # cannot exist until Saturday evening. Writing one into a write-once file would
    # permanently foreclose the grid-conditional forecast this project actually ships, and
    # put a different kind of prediction into the same track record — R11 was scored
    # against its qualifying classification, so a gridless R12 would not be comparable.
    #
    # Revisit when a qualifying model exists: at that point a pre-weekend forecast is the
    # product rather than a placeholder, and this guard becomes the wrong behaviour.
    if diag.get("grid_source") in (None, "none"):
        print("  forecast is grid-free — prediction log deferred until qualifying")
        return

    log_dir = WEB_DATA / "predictions"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{season}_R{int(rnd):02d}.json"
    if path.exists():
        print(f"  prediction log already exists for R{int(rnd)} — left untouched")
        return

    # Only what is needed to score the forecast later, so the log stays small enough to
    # commit for every round of a season.
    entry = {
        "schema_version": 1,
        "season": season,
        "round": int(rnd),
        "generated_utc": payload["generated_utc"],
        "circuit": diag.get("circuit"),
        "grid_source": diag.get("grid_source"),
        "likelihood": diag.get("likelihood"),
        "forecast": payload["forecast"],
        "position_matrix": payload["position_matrix"],
        "weather": payload.get("weather"),
    }
    path.write_text(json.dumps(entry, indent=1))
    print(f"  prediction log: {path.name}  (grid: {diag.get('grid_source')})")


def export_season(season: int) -> None:
    """Year so far: every race run, the podium, and both championship tables.

    Straight results, no model output. The dashboard is otherwise wall-to-wall inference,
    and a reader who wants to know what has actually happened this season should not have
    to infer it from a forecast. It doubles as the check on everything else: if the
    standings here disagree with the strength model's ordering, one of them is wrong.

    Where a forecast was published for a race, its scored result is attached from the
    prediction log, so the track record sits next to the results rather than in a report
    nobody opens.
    """
    files = sorted(glob.glob(str(RAW / f"results_{season}_R*_R.parquet")))
    if not files:
        print("no race results to export")
        return
    res = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    res["Position"] = pd.to_numeric(res["Position"], errors="coerce")
    res["Points"] = pd.to_numeric(res["Points"], errors="coerce").fillna(0.0)
    res["GridPosition"] = pd.to_numeric(res["GridPosition"], errors="coerce")
    res["classified"] = res["ClassifiedPosition"].astype(str).str.isdigit()

    sfiles = sorted(glob.glob(str(RAW / f"results_{season}_R*_S.parquet")))
    sprints = (pd.concat([pd.read_parquet(f) for f in sfiles], ignore_index=True)
               if sfiles else pd.DataFrame())
    if not sprints.empty:
        sprints["Points"] = pd.to_numeric(sprints["Points"], errors="coerce").fillna(0.0)

    scored = {}
    for p in sorted((WEB_DATA / "predictions").glob(f"{season}_R*.json")) \
            if (WEB_DATA / "predictions").exists() else []:
        try:
            log = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        fc = {f["driver"]: f for f in log.get("forecast", [])}
        rnd = int(log["round"])
        got = res[(res["round"] == rnd) & (res["Position"] == 1)]
        if got.empty or not fc:
            continue
        winner = str(got["Abbreviation"].iloc[0])
        fav = max(fc.values(), key=lambda f: f.get("p_win") or 0.0)
        scored[rnd] = {
            "favourite": fav["driver"],
            "p_win": round(float(fav.get("p_win") or 0.0), 4),
            "hit": fav["driver"] == winner,
            "p_win_of_actual_winner": round(float((fc.get(winner) or {}).get("p_win") or 0.0), 4),
            "grid_source": log.get("grid_source"),
        }

    races = []
    for (rnd, event), g in res.groupby(["round", "event"], sort=True):
        g = g.sort_values("Position")
        podium = [{"pos": int(r.Position), "driver": r.Abbreviation, "team": r.TeamName}
                  for r in g[g["Position"] <= 3].itertuples()]
        pole = g[g["GridPosition"] == 1]
        # Full classification, so any past round can be opened rather than showing only its
        # podium. Twenty-two rows a race is small enough to ship in the same payload.
        results = [{
            "pos": None if pd.isna(r.Position) else int(r.Position),
            "driver": r.Abbreviation, "team": r.TeamName,
            "grid": None if pd.isna(r.GridPosition) else int(r.GridPosition),
            "points": float(r.Points),
            "status": str(r.Status),
            "classified": bool(r.classified),
        } for r in g.itertuples()]

        races.append({
            "round": int(rnd),
            "event": event,
            "podium": podium,
            "pole": str(pole["Abbreviation"].iloc[0]) if not pole.empty else None,
            "retirements": int((~g["classified"]).sum()),
            "entries": len(g),
            "sprint": bool(not sprints.empty and (sprints["round"] == rnd).any()),
            "forecast": scored.get(int(rnd)),
            "results": results,
        })

    all_pts = pd.concat([res[["Abbreviation", "TeamName", "Points"]],
                         sprints[["Abbreviation", "TeamName", "Points"]]
                         if not sprints.empty else pd.DataFrame()], ignore_index=True)

    wins = res[res["Position"] == 1]["Abbreviation"].value_counts()
    poles = res[res["GridPosition"] == 1]["Abbreviation"].value_counts()
    pods = res[res["Position"] <= 3]["Abbreviation"].value_counts()
    drv = (all_pts.groupby(["Abbreviation", "TeamName"], as_index=False)["Points"].sum()
                  .sort_values("Points", ascending=False))
    drivers = [{"driver": r.Abbreviation, "team": r.TeamName, "points": float(r.Points),
                "wins": int(wins.get(r.Abbreviation, 0)),
                "podiums": int(pods.get(r.Abbreviation, 0)),
                "poles": int(poles.get(r.Abbreviation, 0))}
               for r in drv.itertuples()]

    con = (all_pts.groupby("TeamName", as_index=False)["Points"].sum()
                  .sort_values("Points", ascending=False))
    constructors = [{"team": r.TeamName, "points": float(r.Points)} for r in con.itertuples()]

    # --- how the most recently scored forecast actually did --------------------------
    # `score_race.py` writes these but nothing ever read them back, so the one test that
    # cannot be gamed lived only in a CSV. It belongs on the page.
    last_scored = None
    score_files = sorted(glob.glob(str(REPORTS / f"race_score_{season}_R*.csv")))
    if score_files:
        m = re.search(rf"race_score_{season}_R(\d+)\.csv", score_files[-1])
        rnd = int(m.group(1)) if m else None
        try:
            sc = pd.read_csv(score_files[-1])
        except OSError:
            sc = pd.DataFrame()
        if rnd is not None and not sc.empty:
            ev = next((r["event"] for r in races if r["round"] == rnd), f"Round {rnd}")
            def _row(scope, model):
                q = sc[(sc["scored"] == scope) & (sc["model"] == model)]
                return None if q.empty else q.iloc[0].to_dict()
            fc_full = _row("full field", "published forecast")
            bl_full = _row("full field", "baseline: actual grid")
            last_scored = {
                "round": rnd, "event": ev,
                "forecast": clean([fc_full])[0] if fc_full else None,
                "baseline": clean([bl_full])[0] if bl_full else None,
                "beat_baseline": bool(fc_full and bl_full and fc_full["rps"] < bl_full["rps"]),
                "call": scored.get(rnd),
            }

    hits = [v for v in scored.values()]
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season": season,
        "rounds_complete": int(res["round"].nunique()),
        "races": races,
        "drivers": drivers,
        "constructors": constructors,
        "track_record": {
            "scored": len(hits),
            "winner_called": sum(1 for h in hits if h["hit"]),
        },
        "last_scored": last_scored,
    }
    out = WEB_DATA / f"season_{season}.json"
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}  ({len(races)} races, {len(drivers)} drivers, "
          f"{len(hits)} scored forecasts)")


def export_teams(season: int) -> None:
    import glob

    from apex.teams import build_profiles

    files = sorted(glob.glob(str(RAW / f"results_{season}_R*_R.parquet")))
    results = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    sprint_files = sorted(glob.glob(str(RAW / f"results_{season}_R*_S.parquet")))
    sprint = (pd.concat([pd.read_parquet(f) for f in sprint_files], ignore_index=True)
              if sprint_files else None)
    pace = pd.read_parquet(PROCESSED / f"pace_{season}.parquet")
    cons = pd.read_parquet(PROCESSED / f"constructor_{season}.parquet")
    skill = pd.read_parquet(PROCESSED / f"skill_{season}.parquet")

    profiles = build_profiles(results, pace, cons, skill, sprint_2026=sprint)
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season": season,
        "teams": profiles,
    }
    out = WEB_DATA / f"teams_{season}.json"
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB) · {len(profiles)} teams")


if __name__ == "__main__":
    raise SystemExit(main())
