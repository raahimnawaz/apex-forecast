"""Historical race results from the Jolpica-F1 API (the Ergast successor).

Only used for seasons we do not ingest through FastF1. Jolpica is volunteer-run and
rate-limited, so every response is cached to disk and the dashboard never touches it
at view time.

Why we want 2025 at all, given that 2026 reset the regulations: driver skill carries
across a rule change, car performance does not. Layer 1 uses the older season to
anchor the driver terms while deliberately severing the constructor terms at the
regulation boundary.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from apex.paths import RAW

BASE = "https://api.jolpi.ca/ergast/f1"
PAGE = 100
UA = {"User-Agent": "apex-forecast/0.1 (personal research project)"}


def _get(url: str, params: dict, tries: int = 4) -> dict:
    """GET with backoff. Jolpica returns 429 under load and asks us to slow down."""
    for attempt in range(tries):
        r = requests.get(url, params=params, headers=UA, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
    raise RuntimeError(f"Jolpica request failed after {tries} tries: {url} {params}")


def fetch_season_results(season: int, force: bool = False) -> pd.DataFrame:
    """Every race result for a season, one row per driver-race."""
    cache = RAW / f"jolpica_results_{season}.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    rows, offset = [], 0
    while True:
        data = _get(f"{BASE}/{season}/results/", {"limit": PAGE, "offset": offset})
        table = data["MRData"]["RaceTable"]["Races"]
        for race in table:
            rnd = int(race["round"])
            for res in race["Results"]:
                status = res["status"]
                rows.append({
                    "season": season,
                    "round": rnd,
                    "event": race["raceName"],
                    "circuit": race["Circuit"]["circuitId"],
                    "date": race.get("date"),
                    "driver_id": res["Driver"]["driverId"],
                    "code": res["Driver"].get("code") or res["Driver"]["driverId"][:3].upper(),
                    "given_name": res["Driver"]["givenName"],
                    "family_name": res["Driver"]["familyName"],
                    "constructor_id": res["Constructor"]["constructorId"],
                    "constructor": res["Constructor"]["name"],
                    "grid": int(res["grid"]),
                    "position": int(res["position"]),
                    "position_text": res["positionText"],
                    "points": float(res["points"]),
                    "status": status,
                    # Ergast/Jolpica encodes classification in positionText: a digit means
                    # classified, "R" retired, "D" disqualified, "W" withdrawn, "E"/"F" etc.
                    "classified": res["positionText"].isdigit(),
                    "finished": status == "Finished" or status.startswith("+"),
                })
        total = int(data["MRData"]["total"])
        offset += PAGE
        if offset >= total:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"no results returned for {season}")
    df.to_parquet(cache, index=False)
    return df
