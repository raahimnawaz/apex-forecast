"""Phase 0 spike: verify FastF1 3.8.3 can read a 2026 session before we build a pipeline on it.

FastF1 3.8.3 was released April 2025, before most of the 2026 season ran. The live-timing
API is stable across seasons, but team/driver constants and any schema drift are the risk.
This script fails loudly rather than letting bad data leak into the feature store.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fastf1
import pandas as pd

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))

    print(f"fastf1 {fastf1.__version__}\n")

    # 1. Can we see the 2026 calendar at all?
    sched = fastf1.get_event_schedule(2026, include_testing=False)
    print(f"2026 calendar: {len(sched)} rounds")
    print(sched[["RoundNumber", "EventName", "EventDate", "EventFormat"]].to_string(index=False))
    print()

    # 2. Load the most recent completed race and inspect what actually came back.
    today = pd.Timestamp.now().normalize()
    completed = sched[sched["EventDate"] < today]
    if completed.empty:
        print("FAIL: no completed 2026 events found")
        return 1

    ev = completed.iloc[-1]
    rnd = int(ev["RoundNumber"])
    print(f"Loading R{rnd} {ev['EventName']} — Race\n")

    ses = fastf1.get_session(2026, rnd, "R")
    ses.load(laps=True, telemetry=False, weather=True, messages=True)

    laps = ses.laps
    res = ses.results

    print(f"results rows : {len(res)}")
    print(f"laps rows    : {len(laps)}")
    print(f"drivers      : {laps['Driver'].nunique()}")
    print(f"teams        : {sorted(res['TeamName'].unique().tolist())}")
    print()

    # 3. The columns Layer 0 depends on must exist and be populated.
    required = ["LapTime", "LapNumber", "Compound", "TyreLife", "Stint", "Driver", "Team",
                "TrackStatus", "IsAccurate", "PitInTime", "PitOutTime"]
    missing = [c for c in required if c not in laps.columns]
    if missing:
        print(f"FAIL: missing lap columns: {missing}")
        return 1

    print("column population on laps:")
    for c in required:
        nn = laps[c].notna().sum()
        print(f"  {c:<12} {nn:>5}/{len(laps)}  ({100 * nn / max(len(laps), 1):.0f}%)")
    print()

    # 4. Compound labels — 2026 runs narrower tyres; confirm the labels aren't garbage.
    print("compounds:", laps["Compound"].value_counts(dropna=False).to_dict())
    print("weather cols:", list(ses.weather_data.columns) if ses.weather_data is not None else None)
    print()

    # 5. Sanity: median green-flag lap time should be plausible (60-120s at any real circuit).
    clean = laps[laps["IsAccurate"] & laps["LapTime"].notna()]
    if clean.empty:
        print("FAIL: zero accurate laps — schema or parsing problem")
        return 1
    med = clean["LapTime"].dt.total_seconds().median()
    print(f"median accurate lap: {med:.3f}s over {len(clean)} laps")
    if not 55 < med < 130:
        print(f"FAIL: implausible median lap time {med:.1f}s")
        return 1

    # 6. Phantom-lap check (the 3.8.3 fix): drivers in laps but not in results.
    ghosts = set(laps["Driver"].unique()) - set(res["Abbreviation"].unique())
    print(f"drivers in laps but not results: {ghosts or 'none'}")

    print("\nPASS — 2026 session data is usable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
