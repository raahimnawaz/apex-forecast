"""Where the season is, and the one thing to do next.

The pipeline is six make targets with an implicit order and two manual steps that are easy
to forget — and the expensive failure is silent. Forget `grids/{season}_R{n}.csv` and the
build quietly forecasts off the qualifying classification, which is how six cars were
wrong at the Hungarian Grand Prix.

All of that state is already on disk. This reads it and says what to do, so the order does
not have to live in anyone's head. Nothing here fits a model or writes a file; it is safe
to run at any time and answers in about a second.
"""

from __future__ import annotations

import argparse
import glob
import json
import re

import pandas as pd

from apex.paths import GRIDS, PROCESSED, RAW, REPORTS, WEB_DATA

TICK, CROSS, DOT = "✓", "!", "·"


def ingested_rounds(season: int) -> list[int]:
    out = []
    for f in glob.glob(str(RAW / f"results_{season}_R*_R.parquet")):
        m = re.search(rf"results_{season}_R(\d+)_R\.parquet", f)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def schedule(season: int) -> pd.DataFrame | None:
    """The calendar, from FastF1's cache. Returns None rather than raising — a status
    command that fails when the network is down would be worse than one that says less."""
    try:
        import fastf1

        from apex.paths import CACHE
        fastf1.Cache.enable_cache(str(CACHE))
        return fastf1.get_event_schedule(season, include_testing=False)
    except Exception:  # noqa: BLE001
        return None


def published_round(season: int) -> int | None:
    p = WEB_DATA / f"strength_{season}.json"
    if not p.exists():
        return None
    try:
        return (json.loads(p.read_text()).get("diagnostics") or {}).get("forecast_round")
    except (OSError, ValueError):
        return None


def logged_rounds(season: int) -> set[int]:
    d = WEB_DATA / "predictions"
    if not d.exists():
        return set()
    out = set()
    for f in d.glob(f"{season}_R*.json"):
        m = re.search(rf"{season}_R(\d+)\.json", f.name)
        if m:
            out.add(int(m.group(1)))
    return out


def scored_rounds(season: int) -> set[int]:
    out = set()
    for f in glob.glob(str(REPORTS / f"race_score_{season}_R*.csv")):
        m = re.search(rf"race_score_{season}_R(\d+)\.csv", f)
        if m:
            out.add(int(m.group(1)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    args = ap.parse_args()
    season = args.season

    rounds = ingested_rounds(season)
    if not rounds:
        print("No race data at all.\n\n  Next:  make data")
        return 0

    last = max(rounds)
    nxt = last + 1
    sched = schedule(season)
    total = int(sched["RoundNumber"].max()) if sched is not None else 22

    name, days = None, None
    if sched is not None:
        row = sched[sched["RoundNumber"] == nxt]
        if not row.empty:
            name = str(row["EventName"].iloc[0])
            d = pd.Timestamp(row["EventDate"].iloc[0]).normalize()
            days = int((d - pd.Timestamp.now().normalize()).days)

    quali = (RAW / f"results_{season}_R{nxt:02d}_Q.parquet").exists()
    race_done = (RAW / f"results_{season}_R{nxt:02d}_R.parquet").exists()
    grid_file = GRIDS / f"{season}_R{nxt:02d}.csv"
    pub = published_round(season)
    logs = logged_rounds(season)
    scored = scored_rounds(season)
    fitted = (PROCESSED / f"skill_{season}.parquet").exists()

    when = ""
    if days is not None:
        when = ("today" if days == 0 else "tomorrow" if days == 1
                else f"in {days} days" if days > 0 else f"{-days} days ago")
    head = f"Round {last} of {total} complete"
    if name:
        head += f"  {DOT}  next: R{nxt} {name} {when}"
    print(head)
    print("-" * max(len(head), 46))

    def line(ok, text):
        print(f"  {TICK if ok else CROSS} {text}")

    line(fitted, "model fitted" if fitted else "model not fitted yet")
    line(pub == nxt,
         f"forecast published for R{nxt}" if pub == nxt
         else f"published forecast is for R{pub}, not R{nxt}")
    line(quali, f"R{nxt} qualifying ingested" if quali else f"R{nxt} qualifying not run yet")
    if quali:
        line(grid_file.exists(),
             f"grid corrected for penalties ({grid_file.name})" if grid_file.exists()
             else f"NO {grid_file.name} — forecast would use the qualifying order")

    # Anything raced but never scored is a track record going unrecorded.
    unscored = sorted(r for r in rounds if r in logs and r not in scored)
    missing_log = sorted(r for r in rounds if r not in logs)
    if unscored:
        line(False, f"logged but never scored: R{', R'.join(map(str, unscored))}")
    if missing_log:
        print(f"  {DOT} no prediction log for R{', R'.join(map(str, missing_log))} "
              f"(published before the log existed; they cannot be scored)")

    # ---- the single next action -------------------------------------------------------
    print()
    if unscored:
        r = unscored[0]
        print(f"  Next:  python scripts/score_race.py --round {r}")
        print(f"         R{r} has a forecast logged before the race and no score yet.")
    elif race_done:
        print("  Next:  make all")
        print(f"         R{nxt} has run — ingest it, refit, and forecast R{nxt + 1}.")
    elif quali and not grid_file.exists():
        print(f"  Next:  write {grid_file}  (driver,grid), then: make strength web")
        print("         Penalties are public now and in no feed this project can read.")
        print("         Skipping this is the error that cost the R11 forecast six cars.")
    elif quali:
        print("  Next:  make strength web")
        print(f"         Grid is known and corrected — forecast R{nxt} conditional on it.")
    elif pub == nxt:
        print("  Next:  nothing. Waiting on qualifying.")
        print(f"         The R{nxt} forecast is published and grid-free, which is correct")
        print("         until qualifying runs. `make news` refreshes headlines any time.")
    else:
        print("  Next:  make all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
