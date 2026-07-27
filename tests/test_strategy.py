"""Tests for the pit strategy layer.

The optimiser is the part worth pinning down hardest: it is exact, so its answers are
checkable against hand-computed alternatives rather than merely plausible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apex.strategy import (
    MIN_STINT,
    TyreModel,
    _stint_cost,
    measure_pit_loss,
    optimise_plan,
    season_average_profile,
    seconds_to_theta,
    theta_offsets,
    with_circuit_profile,
)

# --------------------------------------------------------------------------------------
# the optimiser
# --------------------------------------------------------------------------------------

def test_stint_cost_is_the_arithmetic_series():
    # A 5-lap stint at 0.1 s/lap degradation ages 0+1+2+3+4 = 10 lap-units.
    assert _stint_cost(5, 0.1, 0.0) == pytest.approx(1.0)
    # The compound offset is charged on every lap of the stint.
    assert _stint_cost(5, 0.0, 0.2) == pytest.approx(1.0)


def test_high_degradation_buys_more_stops():
    """The core trade: degradation is paid quadratically in stint length, pit loss linearly."""
    deg_lo = {"SOFT": 0.005, "HARD": 0.005}
    deg_hi = {"SOFT": 0.20, "HARD": 0.20}
    off = {"SOFT": 0.0, "HARD": 0.0}
    lo = optimise_plan(60, 22.0, deg_lo, off)
    hi = optimise_plan(60, 22.0, deg_hi, off)
    assert lo is not None and hi is not None
    assert hi.n_stops > lo.n_stops


def test_expensive_pit_lane_buys_fewer_stops():
    deg = {"SOFT": 0.08, "HARD": 0.08}
    off = {"SOFT": 0.0, "HARD": 0.0}
    cheap = optimise_plan(60, 5.0, deg, off)
    dear = optimise_plan(60, 60.0, deg, off)
    assert cheap is not None and dear is not None
    assert cheap.n_stops >= dear.n_stops


def test_plan_is_a_valid_race():
    deg = {"SOFT": 0.12, "MEDIUM": 0.06, "HARD": 0.03}
    off = {"SOFT": -0.4, "MEDIUM": 0.0, "HARD": 0.3}
    p = optimise_plan(66, 24.0, deg, off)
    assert p is not None
    assert sum(p.stints) == 66                       # the whole race distance is covered
    assert len(p.stints) == len(p.compounds) == p.n_stops + 1
    assert min(p.stints) >= MIN_STINT
    assert len(set(p.compounds)) >= 2                # the two-compound rule


def test_optimum_really_is_optimal():
    """Exhaustively beat the DP against brute force on a small instance."""
    deg = {"SOFT": 0.15, "HARD": 0.05}
    off = {"SOFT": -0.3, "HARD": 0.2}
    n_laps, pit = 40, 20.0
    p = optimise_plan(n_laps, pit, deg, off, max_stops=2)
    assert p is not None

    best = np.inf
    for a in range(MIN_STINT, n_laps - MIN_STINT + 1):
        for ca in ("SOFT", "HARD"):
            for cb in ("SOFT", "HARD"):
                if len({ca, cb}) < 2:
                    continue
                b = n_laps - a
                if b < MIN_STINT:
                    continue
                best = min(best, _stint_cost(a, deg[ca], off[ca])
                           + _stint_cost(b, deg[cb], off[cb]) + pit)
    assert p.total_s <= best + 1e-9


def test_max_stint_cap_is_respected():
    deg = {"SOFT": 0.001, "HARD": 0.001}      # so low the optimum wants one long stint
    off = {"SOFT": 0.0, "HARD": 0.0}
    p = optimise_plan(70, 25.0, deg, off, max_stint=20)
    assert p is not None
    assert max(p.stints) <= 20


def test_two_compound_rule_makes_a_zero_stop_illegal():
    deg = {"SOFT": 0.0, "HARD": 0.0}
    off = {"SOFT": 0.0, "HARD": 0.0}
    p = optimise_plan(50, 25.0, deg, off)
    assert p is not None
    assert p.n_stops >= 1                      # never free, even with no degradation at all


def test_single_compound_circuit_has_no_legal_plan():
    p = optimise_plan(50, 22.0, {"SOFT": 0.05}, {"SOFT": 0.0})
    assert p is None


# --------------------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------------------

def _laps_with_stop(loss_s: float, sc: bool = False) -> pd.DataFrame:
    """One driver, 10 green laps of 90 s, with a stop costing `loss_s` split over in/out."""
    rows = []
    for lap in range(1, 11):
        pit_in, pit_out = lap == 5, lap == 6
        t = 90.0 + (loss_s / 2 if pit_in or pit_out else 0.0)
        rows.append({
            "season": 2026, "round": 1, "event": "Test GP", "session": "R",
            "Driver": "AAA", "Team": "T", "LapNumber": lap, "Stint": 1 if lap < 6 else 2,
            "Compound": "MEDIUM", "TyreLife": lap, "lap_time_s": t,
            "IsAccurate": True, "is_green": not (sc and pit_in), "is_pit_in": pit_in,
            "is_pit_out": pit_out,
        })
    return pd.DataFrame(rows)


def test_pit_loss_recovers_a_known_stop_cost():
    got = measure_pit_loss(_laps_with_stop(24.0), 2026)
    assert len(got) == 1
    assert got["pit_loss_raw"].iloc[0] == pytest.approx(24.0, abs=0.5)


def test_safety_car_stops_are_excluded():
    """A stop under a safety car is roughly half price and would bias the estimate."""
    assert measure_pit_loss(_laps_with_stop(24.0, sc=True), 2026).empty


def test_pit_loss_is_shrunk_toward_the_season_median():
    """A circuit with few measured stops must not assert an extreme pit loss."""
    a = _laps_with_stop(22.0)
    b = _laps_with_stop(60.0)                  # one wild stop at another circuit
    b["round"], b["event"] = 2, "Outlier GP"
    got = measure_pit_loss(pd.concat([a, b], ignore_index=True), 2026)
    row = got[got["event"] == "Outlier GP"].iloc[0]
    assert row["pit_loss_s"] < row["pit_loss_raw"]


# --------------------------------------------------------------------------------------
# the offsets
# --------------------------------------------------------------------------------------

def _tyre_model(team_offsets: dict[str, float], events=("A GP", "B GP")) -> TyreModel:
    circ = pd.DataFrame([
        {"season": 2026, "round": i + 1, "event": e, "compound": c,
         "deg_s_per_lap": d, "offset_s": 0.0, "n_laps": 300}
        for i, (e, lo) in enumerate(zip(events, (0.03, 0.12)))
        for c, d in (("SOFT", lo * 1.5), ("MEDIUM", lo), ("HARD", lo * 0.6))
    ])
    team = pd.DataFrame([{"team": t, "deg_offset_s": v, "se": 0.003,
                          "n_races": 10, "shrunk": v} for t, v in team_offsets.items()])
    laps = pd.DataFrame([{"round": i + 1, "event": e, "n_laps": 60, "max_stint": 30}
                         for i, e in enumerate(events)])
    pit = pd.DataFrame([{"event": e, "round": i + 1, "stops": 20, "pit_loss_s": 22.0,
                         "pit_loss_raw": 22.0} for i, e in enumerate(events)])
    return TyreModel(by_circuit=circ, by_team=team, pit_loss=pit, race_laps=laps,
                     shrinkage=0.8)


def test_offsets_are_centred_and_so_cannot_shift_the_field():
    """A constant added to every theta is invisible to a Plackett-Luce; it must be removed."""
    t = _tyre_model({"good": -0.02, "mid": 0.0, "bad": 0.02})
    off = theta_offsets(t, "A GP", ["good", "mid", "bad"], scale=1.7)
    assert off.sum() == pytest.approx(0.0, abs=1e-9)


def test_identical_teams_get_identical_offsets():
    t = _tyre_model({"x": 0.0, "y": 0.0, "z": 0.0})
    off = theta_offsets(t, "A GP", ["x", "y", "z"], scale=1.7)
    assert np.allclose(off, 0.0, atol=1e-9)


def test_a_zero_scale_switches_the_layer_off():
    t = _tyre_model({"good": -0.02, "bad": 0.02})
    assert np.allclose(theta_offsets(t, "A GP", ["good", "bad"], scale=0.0), 0.0)


def test_seconds_to_theta_refuses_a_broken_fit():
    """A positive slope would mean slower cars rank higher — a broken fit, not a finding."""
    pace = pd.DataFrame({"Team": ["a", "b", "c", "d"], "pace_s": [-1.0, 0.0, 0.5, 1.0]})
    good = pd.DataFrame({"constructor": ["a", "b", "c", "d"],
                         "car_2026_latest": [3.0, 1.0, 0.0, -2.0]})
    bad = good.assign(car_2026_latest=lambda x: -x["car_2026_latest"])
    assert seconds_to_theta(pace, good) > 0
    assert seconds_to_theta(pace, bad) == 0.0


# --------------------------------------------------------------------------------------
# the leak-free path
# --------------------------------------------------------------------------------------

def test_season_average_profile_excludes_the_target_circuit():
    """The forecast circuit's own race is the thing being predicted; it cannot inform itself."""
    t = _tyre_model({"x": 0.0}, events=("A GP", "B GP"))
    prof = season_average_profile(t, "B GP", rnd=2)
    a_only = t.by_circuit[t.by_circuit["event"] == "A GP"].set_index("compound")
    for r in prof.itertuples():
        assert r.deg_s_per_lap == pytest.approx(a_only.loc[r.compound, "deg_s_per_lap"])


def test_with_circuit_profile_leaves_team_degradation_alone():
    t = _tyre_model({"good": -0.02, "bad": 0.02})
    swapped = with_circuit_profile(t, season_average_profile(t, "B GP", rnd=2))
    pd.testing.assert_frame_equal(t.by_team, swapped.by_team)
    assert set(swapped.by_circuit["event"]) == {"A GP", "B GP"}
