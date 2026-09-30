"""Tests for choosing which race weekend's qualifying to ingest early.

This window decides whether a round gets a scoreable forecast at all. Too early and the
write-once log records a half-published grid; too late and it records a forecast made
with the result already known. Rounds 12-15 of 2026 went unscored because nothing
opened the window, so both edges are pinned here.
"""

from __future__ import annotations

import pandas as pd

from apex.ingest import QUALI_SETTLE, session_start_utc, weekend_awaiting_race

CONVENTIONAL = ("Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race")
SPRINT = ("Practice 1", "Sprint Qualifying", "Sprint", "Qualifying", "Race")


def _event(rnd: int, names: tuple[str, ...], quali: str, race: str) -> dict:
    ev = {"RoundNumber": rnd}
    q, r = pd.Timestamp(quali), pd.Timestamp(race)
    for i, name in enumerate(names, start=1):
        ev[f"Session{i}"] = name
        ev[f"Session{i}DateUtc"] = {"Qualifying": q, "Race": r}.get(name, q - pd.Timedelta(days=1))
    return ev


# Mirrors the real 2026 schedule around the gap: Baku raced on a Saturday, and the next
# two rounds are a conventional weekend and a sprint weekend.
SCHED = pd.DataFrame([
    _event(15, CONVENTIONAL, "2026-09-25 12:00", "2026-09-26 11:00"),
    _event(16, CONVENTIONAL, "2026-10-03 08:00", "2026-10-04 07:00"),
    _event(17, SPRINT, "2026-10-10 13:00", "2026-10-11 12:00"),
])


def test_session_start_is_found_by_name_not_position():
    sprint = SCHED.iloc[2]
    assert session_start_utc(sprint, "Qualifying") == pd.Timestamp("2026-10-10 13:00")
    assert session_start_utc(sprint, "Race") == pd.Timestamp("2026-10-11 12:00")
    assert session_start_utc(sprint, "Sprint Shootout") is None


def test_saturday_evening_after_qualifying_takes_that_round():
    # 18:00 PDT, when the Saturday launchd job fires.
    assert weekend_awaiting_race(SCHED, pd.Timestamp("2026-10-04 01:00")) == 16


def test_sprint_weekend_uses_the_grand_prix_qualifying():
    assert weekend_awaiting_race(SCHED, pd.Timestamp("2026-10-11 01:00")) == 17


def test_qualifying_that_has_not_settled_is_left_alone():
    just_after = pd.Timestamp("2026-10-03 08:00") + QUALI_SETTLE - pd.Timedelta(minutes=1)
    assert weekend_awaiting_race(SCHED, just_after) is None
    assert weekend_awaiting_race(SCHED, just_after + pd.Timedelta(minutes=1)) == 16


def test_nothing_once_the_race_has_started():
    """A Saturday race with the job at 18:00 — the Baku case. Too late to forecast."""
    assert weekend_awaiting_race(SCHED, pd.Timestamp("2026-09-27 01:00")) is None
    assert weekend_awaiting_race(SCHED, pd.Timestamp("2026-10-04 07:00")) is None


def test_midweek_has_no_pending_weekend():
    assert weekend_awaiting_race(SCHED, pd.Timestamp("2026-09-29 16:00")) is None
