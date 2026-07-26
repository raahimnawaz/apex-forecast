"""Team profiles: everything the dashboard says about a constructor.

Every figure here is computed from race results or from the two model layers. Nothing
is written from background knowledge except the static reference facts below (power
unit supplier and debut season), which are constants, not judgements.

"Recent form" is deliberately a measurement rather than a narrative: the change in
Layer 0 corrected pace between a team's first three races and its last three, and the
Layer 1 development term over the same window. A team's story this season is whether
its car got quicker, and that is a number we already have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Verified 2026 grid reference data. Power units confirmed against published
# 2026 supplier listings; five manufacturers serve eleven teams.
TEAM_REFERENCE = {
    "Mercedes":        {"power_unit": "Mercedes", "works": True,  "first_season": 2010},
    "Ferrari":         {"power_unit": "Ferrari", "works": True,  "first_season": 1950},
    "Red Bull Racing": {"power_unit": "Red Bull Powertrains", "works": True, "first_season": 2005},
    "McLaren":         {"power_unit": "Mercedes", "works": False, "first_season": 1966},
    "Aston Martin":    {"power_unit": "Honda", "works": False, "first_season": 2021},
    "Alpine":          {"power_unit": "Mercedes", "works": False, "first_season": 2021},
    "Williams":        {"power_unit": "Mercedes", "works": False, "first_season": 1977},
    "Racing Bulls":    {"power_unit": "Red Bull Powertrains", "works": False, "first_season": 2024},
    "Audi":            {"power_unit": "Audi", "works": True,  "first_season": 2026},
    "Haas F1 Team":    {"power_unit": "Ferrari", "works": False, "first_season": 2016},
    "Cadillac":        {"power_unit": "Ferrari", "works": False, "first_season": 2026},
}

# Teams whose entry is a rebrand or a debut deserve an explicit caveat in the UI:
# their model estimates rest on far less evidence than a continuing entrant's.
ENTRY_NOTE = {
    "Audi": "Rebranded from Sauber for 2026 and now a works manufacturer, so pre-2026 "
            "history belongs to a different operation.",
    "Cadillac": "Debut season. No prior F1 history at all, so every estimate for this "
                "team rests on ten races and deserves more scepticism than the rest of "
                "the grid.",
    "Racing Bulls": "Continues the Red Bull junior entry, which has changed name "
                    "repeatedly; only the 2026 car is relevant under the new rules.",
}


def build_profiles(results_2026: pd.DataFrame,
                   pace: pd.DataFrame,
                   constructor: pd.DataFrame,
                   skill: pd.DataFrame,
                   sprint_2026: pd.DataFrame | None = None) -> list[dict]:
    """One record per constructor, combining results with both model layers.

    Championship points must include the sprint races. 2026 runs six sprint weekends
    paying 8-7-6-5-4-3-2-1, so counting only grand prix results understates a team by
    tens of points — Mercedes reads 311 instead of 358. Wins, podiums, retirements and
    best finish stay grand-prix-only, which is their conventional meaning.
    """
    r = results_2026.copy()
    r["Position"] = pd.to_numeric(r["Position"], errors="coerce")
    r["GridPosition"] = pd.to_numeric(r["GridPosition"], errors="coerce")
    r["classified"] = r["ClassifiedPosition"].astype(str).str.isdigit()

    sprint_team = sprint_driver = None
    if sprint_2026 is not None and not sprint_2026.empty:
        s = sprint_2026.copy()
        s["Points"] = pd.to_numeric(s["Points"], errors="coerce").fillna(0.0)
        sprint_team = s.groupby("TeamName")["Points"].sum()
        sprint_driver = s.groupby(["TeamName", "Abbreviation"])["Points"].sum()

    strength = constructor.set_index("constructor")
    pace_team = pace.groupby("Team")
    skill_ix = skill.set_index("driver")
    rounds = sorted(pace["round"].unique())
    early, late = rounds[:3], rounds[-3:]

    profiles = []
    for team, g in r.groupby("TeamName"):
        drivers = []
        for code, dg in g.groupby("Abbreviation"):
            dp = pace[(pace["Team"] == team) & (pace["Driver"] == code)]
            sk = skill_ix.loc[code] if code in skill_ix.index else None
            sp_d = 0.0 if sprint_driver is None else float(sprint_driver.get((team, code), 0.0))
            drivers.append({
                "code": code,
                "name": str(dg["FullName"].iloc[0]),
                "points": float(dg["Points"].sum()) + sp_d,
                "starts": len(dg),
                "wins": int((dg["Position"] == 1).sum()),
                "podiums": int((dg["Position"] <= 3).sum()),
                "best_finish": None if dg[dg["classified"]].empty
                               else int(dg.loc[dg["classified"], "Position"].min()),
                "dnf": int((~dg["classified"]).sum()),
                "avg_grid": None if dg["GridPosition"].isna().all()
                            else round(float(dg["GridPosition"].mean()), 2),
                "pace_s": None if dp.empty else round(float(dp["pace_s"].mean()), 4),
                "skill": None if sk is None else round(float(sk["skill"]), 4),
            })
        drivers.sort(key=lambda d: -d["points"])

        tp = pace_team.get_group(team) if team in pace_team.groups else pd.DataFrame()
        early_pace = tp[tp["round"].isin(early)]["pace_s"].mean() if not tp.empty else np.nan
        late_pace = tp[tp["round"].isin(late)]["pace_s"].mean() if not tp.empty else np.nan
        # Negative pace is quicker, so an improving team has a NEGATIVE delta.
        pace_delta = (late_pace - early_pace) if not (np.isnan(early_pace) or np.isnan(late_pace)) else None

        s = strength.loc[team] if team in strength.index else None

        ref = TEAM_REFERENCE.get(team, {})
        profiles.append({
            "team": team,
            "power_unit": ref.get("power_unit"),
            "works": ref.get("works"),
            "first_season": ref.get("first_season"),
            "entry_note": ENTRY_NOTE.get(team),
            "points": float(g["Points"].sum()) + (0.0 if sprint_team is None
                                                  else float(sprint_team.get(team, 0.0))),
            "wins": int((g["Position"] == 1).sum()),
            "podiums": int((g["Position"] <= 3).sum()),
            "dnf": int((~g["classified"]).sum()),
            "starts": len(g),
            "best_finish": None if g[g["classified"]].empty
                           else int(g.loc[g["classified"], "Position"].min()),
            "avg_grid": None if g["GridPosition"].isna().all()
                        else round(float(g["GridPosition"].mean()), 2),
            "pace_s": None if tp.empty else round(float(tp["pace_s"].mean()), 4),
            "pace_delta_s": None if pace_delta is None else round(float(pace_delta), 4),
            "strength": None if s is None else round(float(s["car_2026_latest"]), 4),
            "development": None if s is None else round(float(s["development"]), 4),
            "drivers": drivers,
        })

    profiles.sort(key=lambda p: -p["points"])
    for i, p in enumerate(profiles):
        p["championship_position"] = i + 1
    return profiles
