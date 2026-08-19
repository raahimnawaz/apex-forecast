"""Phase 0 spike: verify FastF1 3.8.3 can read a 2026 session before we build a pipeline on it.

FastF1 3.8.3 was released April 2025, before most of the 2026 season ran. The live-timing
API is stable across seasons, but team/driver constants and any schema drift are the risk.
This script fails loudly rather than letting bad data leak into the feature store.

There are two ways it can fail and they mean opposite things:

    exit 1  the data arrived and is wrong — schema drift or a stale pin, a real finding
    exit 2  the data never arrived — this machine is not allowed to read the feed, which
            says nothing about the data or the pin

Separating them matters, because the second is not a defect in this project.
`livetiming.formula1.com` answers 403 to datacenter IPs, and the mirror FastF1 falls back
to does not carry archived sessions, so it answers 404. Measured 2026-08-19 against the
R11 race feed: 403 then 404 from a GitHub-hosted runner, 200 and 7.2 MB from a residential
connection, on the same pin and the same session. That is why the refresh runs locally
rather than in Actions — see .github/workflows/update.yml.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fastf1
import pandas as pd
import requests

# Private module, but the pin is exact and these are the two hosts the library itself
# requests, in this order. Reading them from the library keeps the probe honest rather
# than hardcoding URLs that could drift from what FastF1 actually asks for.
from fastf1 import _api as ff1_api
from fastf1.exceptions import DataNotLoadedError

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"


def probe_hosts(api_path: str) -> list[tuple[str, str, bool]]:
    """Ask each live-timing host for one real page and report what it said.

    Only called after a load has already failed, to tell "the feed changed" apart from
    "this machine cannot read the feed."
    """
    page = ff1_api.pages["timing_data"]
    out: list[tuple[str, str, bool]] = []
    for base in (ff1_api.base_url, ff1_api.base_url_mirror):
        try:
            r = requests.get(base + api_path + page, headers=ff1_api.headers, timeout=45)
            out.append((base, f"HTTP {r.status_code}", r.status_code == 200))
        except requests.RequestException as exc:
            out.append((base, type(exc).__name__, False))
    return out


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
    try:
        ses.load(laps=True, telemetry=False, weather=True, messages=True)
        laps = ses.laps
        res = ses.results
    except (DataNotLoadedError, requests.RequestException) as exc:
        # FastF1 downgrades every failed fetch to a warning and only raises when something
        # reads the empty result, so the exception itself says nothing about the cause.
        # Ask the hosts directly instead of guessing. Anything outside these two is a
        # genuine crash and should surface as one rather than be tidied into a message.
        print(f"load failed: {type(exc).__name__}: {exc}\n")
        print("asking each live-timing host directly:")
        served = False
        for base, status, ok in probe_hosts(ses.api_path):
            print(f"  {base:<42} {status}")
            served = served or ok
        if served:
            print("\nFAIL: a host served the page but the session still would not load.")
            print("That is schema drift or a stale pin — a real finding. Investigate.")
            return 1
        print("\nSKIP: no live-timing host would serve this session to this machine.")
        print("The primary answers 403 to datacenter IPs and the mirror does not carry")
        print("archived sessions. That is an environment fact, not a data problem —")
        print("run the refresh from a normal connection with `make all`.")
        return 2

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
