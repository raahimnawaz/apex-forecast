"""Serialise the model outputs into the versioned JSON the static dashboard reads.

The dashboard never touches a database or an upstream API at view time — it loads
these files and nothing else. That keeps it deployable to static hosting and keeps
Jolpica/FastF1 out of the request path.
"""

from __future__ import annotations

import argparse
import json
import math
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
                     best_s=("pace_s", "min"),
                     worst_s=("pace_s", "max"),
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
        "degradation_by_round": clean(deg.round(4).to_dict(orient="records")),
        "event_fits": clean(fit.round(4).to_dict(orient="records")),
        "totals": {
            "laps_modelled": int(fit["n_laps"].sum()),
            "races_fitted": len(fit),
            "median_pseudo_r2": float(fit["pseudo_r2"].median()),
            "mean_resid_sd_s": float(fit["resid_sd_s"].mean()),
            "mean_dirty_air_cost_s": float(fit["dirty_air_cost_s"].mean()),
        },
    }

    out = WEB_DATA / f"pace_{season}.json"
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    print(f"  drivers: {len(agg)}  rounds: {payload['rounds_analysed']}")
    print(f"  laps modelled: {payload['totals']['laps_modelled']:,}")

    export_strength(season, colors)
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
    per_round = REPORTS / "calibration_walkforward.csv"
    if not summary.exists():
        return None
    s = pd.read_csv(summary)
    rows = clean(s.round(4).to_dict(orient="records"))
    best = min(rows, key=lambda r: r["rps"])["model"]
    grid_rps = next((r["rps"] for r in rows if r["model"] == "baseline: grid"), None)
    return {
        "summary": rows,
        "per_round": clean(pd.read_csv(per_round).round(4).to_dict(orient="records"))
                     if per_round.exists() else [],
        "best_model": best,
        "grid_baseline_rps": grid_rps,
        "beats_grid_baseline": bool(best.startswith("model:")),
        "n_eval_races": int(s["races"].max()),
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
        "calibration": calibration_payload(),
    }
    out = WEB_DATA / f"strength_{season}.json"
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    print(f"  constructor share {100 * diag['constructor_share']:.1f}% · "
          f"R-hat {diag['worst_rhat']} · Layer-0 rho {diag['layer0_spearman']}")


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
    by_round = pd.read_parquet(PROCESSED / "constructor_by_round.parquet")
    skill = pd.read_parquet(PROCESSED / f"skill_{season}.parquet")
    deg = pd.read_parquet(PROCESSED / f"degradation_{season}.parquet")

    profiles = build_profiles(results, pace, cons, by_round, skill, deg, sprint_2026=sprint)
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season": season,
        "rounds_complete": int(results["round"].nunique()),
        "teams": profiles,
    }
    out = WEB_DATA / f"teams_{season}.json"
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB) · {len(profiles)} teams")


if __name__ == "__main__":
    raise SystemExit(main())
