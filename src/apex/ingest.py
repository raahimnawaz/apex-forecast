"""Session ingest: FastF1 -> tidy parquet.

Idempotent. Re-running skips sessions already on disk unless force=True. Every session
also emits a quality record so silently-bad data shows up in the report instead of in
the model.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass

import fastf1
import pandas as pd

from apex.paths import CACHE, RAW

fastf1.Cache.enable_cache(str(CACHE))
warnings.filterwarnings("ignore", category=FutureWarning)

# Track status codes from the F1 live-timing feed. A lap can carry several concatenated.
GREEN, YELLOW, SC, RED, VSC_ON, VSC_OFF = "1", "2", "4", "5", "6", "7"

LAP_COLS = [
    "Driver", "DriverNumber", "Team", "LapNumber", "LapTime", "Stint", "Compound",
    "TyreLife", "FreshTyre", "TrackStatus", "IsAccurate", "Position",
    "Sector1Time", "Sector2Time", "Sector3Time", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
    "PitInTime", "PitOutTime", "LapStartTime", "Time",
]

RESULT_COLS = [
    "DriverNumber", "Abbreviation", "FullName", "TeamName", "Position", "ClassifiedPosition",
    "GridPosition", "Time", "Status", "Points", "Q1", "Q2", "Q3",
]


@dataclass
class Quality:
    """One row per ingested session — the data-quality report is a frame of these."""
    season: int
    round: int
    event: str
    session: str
    n_drivers: int
    n_laps: int
    n_accurate: int
    pct_accurate: float
    n_no_laptime: int
    ghost_drivers: str
    compounds: str
    median_lap_s: float
    ok: bool
    note: str


def _to_seconds(s: pd.Series) -> pd.Series:
    return s.dt.total_seconds() if pd.api.types.is_timedelta64_dtype(s) else s


def _flags(track_status: pd.Series) -> pd.DataFrame:
    """Decompose the concatenated track-status string into boolean flags per lap."""
    ts = track_status.fillna("").astype(str)
    return pd.DataFrame({
        "is_green": ts.eq(GREEN),
        "has_yellow": ts.str.contains(YELLOW),
        "has_sc": ts.str.contains(SC),
        "has_red": ts.str.contains(RED),
        "has_vsc": ts.str.contains(VSC_ON) | ts.str.contains(VSC_OFF),
    }, index=track_status.index)


def ingest_session(season: int, rnd: int, session: str, force: bool = False
                   ) -> tuple[pd.DataFrame | None, Quality | None]:
    """Load one session and persist laps + results. Returns (laps, quality)."""
    tag = f"{season}_R{rnd:02d}_{session}"
    lap_path = RAW / f"laps_{tag}.parquet"
    res_path = RAW / f"results_{tag}.parquet"
    wx_path = RAW / f"weather_{tag}.parquet"

    if lap_path.exists() and not force:
        return pd.read_parquet(lap_path), None

    try:
        ses = fastf1.get_session(season, rnd, session)
        ses.load(laps=True, telemetry=False, weather=True, messages=False)
    except Exception as exc:  # noqa: BLE001 - a failed session must not kill the run
        return None, Quality(season, rnd, "?", session, 0, 0, 0, 0.0, 0, "", "",
                             float("nan"), False, f"load failed: {type(exc).__name__}: {exc}")

    event = str(ses.event["EventName"])
    laps = ses.laps.copy()
    if laps.empty:
        return None, Quality(season, rnd, event, session, 0, 0, 0, 0.0, 0, "", "",
                             float("nan"), False, "no laps returned")

    laps = laps[[c for c in LAP_COLS if c in laps.columns]].copy()
    laps["lap_time_s"] = _to_seconds(laps["LapTime"])
    for c in ("Sector1Time", "Sector2Time", "Sector3Time"):
        if c in laps:
            laps[c.lower().replace("time", "_s")] = _to_seconds(laps[c])
    laps = pd.concat([laps, _flags(laps["TrackStatus"])], axis=1)

    laps["is_pit_in"] = laps["PitInTime"].notna()
    laps["is_pit_out"] = laps["PitOutTime"].notna()

    # Session clock at lap end, in seconds. Needed to reconstruct gaps between cars,
    # which is how we get a dirty-air term into the pace model.
    laps["lap_end_s"] = _to_seconds(laps["Time"])
    laps["lap_start_s"] = _to_seconds(laps["LapStartTime"])

    laps["season"] = season
    laps["round"] = rnd
    laps["session"] = session
    laps["event"] = event

    # Drop raw timedelta columns; parquet round-trips them inconsistently and we have seconds.
    laps = laps.drop(columns=[c for c in ("LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
                                          "PitInTime", "PitOutTime", "LapStartTime", "Time")
                              if c in laps.columns])
    laps.to_parquet(lap_path, index=False)

    res = ses.results
    if res is not None and not res.empty:
        res = res[[c for c in RESULT_COLS if c in res.columns]].copy()
        for c in ("Time", "Q1", "Q2", "Q3"):
            if c in res:
                res[c] = _to_seconds(res[c])
        res["season"], res["round"], res["session"], res["event"] = season, rnd, session, event
        res.to_parquet(res_path, index=False)

    wx = ses.weather_data
    if wx is not None and not wx.empty:
        wx = wx.copy()
        if pd.api.types.is_timedelta64_dtype(wx["Time"]):
            wx["Time"] = wx["Time"].dt.total_seconds()
        wx["season"], wx["round"], wx["session"] = season, rnd, session
        wx.to_parquet(wx_path, index=False)

    acc = laps[laps["IsAccurate"] & laps["lap_time_s"].notna()]
    med = float(acc["lap_time_s"].median()) if not acc.empty else float("nan")
    ghosts = ""
    if res is not None and not res.empty and "Abbreviation" in res:
        ghosts = ",".join(sorted(set(laps["Driver"]) - set(res["Abbreviation"]))) or ""

    has_med = not math.isnan(med)
    q = Quality(
        season=season, round=rnd, event=event, session=session,
        n_drivers=int(laps["Driver"].nunique()), n_laps=len(laps), n_accurate=len(acc),
        pct_accurate=round(100 * len(acc) / len(laps), 1),
        n_no_laptime=int(laps["lap_time_s"].isna().sum()),
        ghost_drivers=ghosts,
        compounds=",".join(sorted(laps["Compound"].dropna().unique())),
        median_lap_s=round(med, 3) if has_med else float("nan"),
        # A plausible median green-flag lap is the cheapest end-to-end check that the
        # session parsed into something real rather than into structurally valid noise.
        ok=bool(len(acc) > 0 and has_med and 40 < med < 200 and not ghosts),
        note="",
    )
    return laps, q


def ingest_season(season: int, sessions=("FP1", "FP2", "FP3", "Q", "R"), through_round: int | None = None,
                  force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ingest every completed round of a season. Sprint weekends are handled by session availability."""
    sched = fastf1.get_event_schedule(season, include_testing=False)
    today = pd.Timestamp.now().normalize()
    sched = sched[sched["EventDate"] < today]
    if through_round is not None:
        sched = sched[sched["RoundNumber"] <= through_round]

    all_laps, quals = [], []
    for _, ev in sched.iterrows():
        rnd = int(ev["RoundNumber"])
        is_sprint = "sprint" in str(ev["EventFormat"]).lower()
        # Sprint weekends replace FP2/FP3 with SQ + Sprint.
        want = ("FP1", "SQ", "S", "Q", "R") if is_sprint else sessions
        for s in want:
            laps, q = ingest_session(season, rnd, s, force=force)
            if laps is not None and not laps.empty:
                all_laps.append(laps)
            if q is not None:
                quals.append(asdict(q))
            print(f"  R{rnd:02d} {ev['EventName'][:28]:<28} {s:<3} "
                  f"{'-' if q is None else ('ok ' if q.ok else 'BAD')} "
                  f"{0 if laps is None else len(laps):>4} laps")

    laps_df = pd.concat(all_laps, ignore_index=True) if all_laps else pd.DataFrame()
    qual_df = pd.DataFrame(quals)
    return laps_df, qual_df
