"""Tests for the reliability and race-simulation layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apex.reliability import Reliability, fit_reliability, simulate_race


def _fake_results():
    """2025 in Jolpica shape, 2026 in FastF1 shape. One team is deliberately fragile."""
    r25 = pd.DataFrame([
        {"code": c, "constructor_id": t, "classified": ok}
        for c, t, reps, ok in [("AAA", "solid", 18, True), ("BBB", "solid", 2, False),
                               ("CCC", "fragile", 10, True), ("DDD", "fragile", 10, False)]
        for _ in range(reps)
    ])
    r26 = pd.DataFrame([
        {"Abbreviation": c, "TeamName": t, "ClassifiedPosition": "1" if ok else "R"}
        for c, t, reps, ok in [("AAA", "solid", 18, True), ("BBB", "solid", 2, False),
                               ("CCC", "fragile", 8, True), ("DDD", "fragile", 12, False)]
        for _ in range(reps)
    ])
    return r25, r26


def _fake_laps(sc_rounds, vsc_rounds, n_rounds=10):
    rows = []
    for rnd in range(1, n_rounds + 1):
        for lap in range(5):
            rows.append({"round": rnd, "event": f"E{rnd}", "session": "R",
                         "has_sc": rnd in sc_rounds and lap == 0,
                         "has_vsc": rnd in vsc_rounds and lap == 0})
    return pd.DataFrame(rows)


def test_fragile_team_gets_higher_retirement_risk():
    r25, r26 = _fake_results()
    rel = fit_reliability(r25, r26, warmup=200, samples=200, chains=1)
    by = rel.by_entry.set_index("driver")
    assert by.loc["DDD", "p_dnf"] > by.loc["AAA", "p_dnf"]
    assert 0.0 < by["p_dnf"].min() and by["p_dnf"].max() < 1.0


def test_estimates_are_shrunk_toward_the_grid_mean():
    """A raw 100% or 0% rate on a handful of starts must not survive as a point estimate."""
    r25, r26 = _fake_results()
    rel = fit_reliability(r25, r26, warmup=200, samples=200, chains=1)
    by = rel.by_entry.set_index("driver")
    # BBB retired every start in both seasons; the estimate must still be below 1.
    assert by.loc["BBB", "raw_rate"] == 1.0
    assert by.loc["BBB", "p_dnf"] < 0.95
    # AAA finished every start; the estimate must stay above 0.
    assert by.loc["AAA", "raw_rate"] == 0.0
    assert by.loc["AAA", "p_dnf"] > 0.02


def test_safety_car_and_vsc_are_counted_separately():
    """The union of SC and VSC fires on every 2026 race, which is why only the full
    safety car is treated as a disruption."""
    r25, r26 = _fake_results()
    laps = _fake_laps(sc_rounds={1, 2, 3}, vsc_rounds={4, 5, 6, 7})
    rel = fit_reliability(r25, r26, laps_2026=laps, warmup=150, samples=150, chains=1)
    assert rel.sc_rate == pytest.approx(0.3, abs=1e-6)
    assert rel.vsc_rate == pytest.approx(0.4, abs=1e-6)


def _theta(n=6, spread=1.0):
    return np.tile(np.linspace(spread, -spread, n), (120, 1))


def test_simulation_returns_a_proper_distribution():
    probs = simulate_race(_theta(), np.full(6, 0.1), 0.3, n_sim=120)
    assert probs.shape == (6, 6)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)


def test_retirement_risk_pushes_a_car_down_the_order():
    """Same pace, worse reliability, worse expected finish — that is the whole point."""
    theta = np.zeros((200, 4))
    p_dnf = np.array([0.02, 0.02, 0.02, 0.60])
    probs = simulate_race(theta, p_dnf, 0.0, n_sim=200, seed=5)
    exp_pos = (probs * np.arange(1, 5)).sum(1)
    assert exp_pos[3] > exp_pos[:3].max()
    assert probs[3, 0] < probs[0, 0]


def test_safety_cars_fatten_the_tail():
    """A back-marker's win probability must rise when safety cars are possible —
    that mechanism is the reason the tail exists at all."""
    theta = _theta(n=6, spread=2.0)
    calm = simulate_race(theta, np.full(6, 0.05), 0.0, n_sim=400, seed=2)
    chaos = simulate_race(theta, np.full(6, 0.05), 0.9, n_sim=400, seed=2)
    assert chaos[-1, 0] > calm[-1, 0]
    assert chaos[0, 0] < calm[0, 0]      # and the favourite gives some back


def test_zero_retirement_risk_reproduces_a_clean_ranking():
    theta = np.tile(np.array([3.0, 2.0, 1.0, 0.0]), (200, 1))
    probs = simulate_race(theta, np.zeros(4), 0.0, n_sim=300, likelihood="forward", seed=1)
    assert probs[:, 0].argmax() == 0
    assert (probs * np.arange(1, 5)).sum(1).argmax() == 3


def test_reliability_dataclass_roundtrip():
    rel = Reliability(by_entry=pd.DataFrame({"driver": ["X"], "p_dnf": [0.1]}),
                      grid_mean=0.14, sc_rate=0.55, vsc_rate=0.64,
                      fcy_by_circuit=pd.DataFrame())
    assert rel.sc_rate < rel.vsc_rate
    assert not rel.by_entry.empty


def test_did_not_start_is_not_a_retirement():
    """A car that never took the start did not retire. Counting it inflates both the
    numerator and the denominator, and a non-start is known before lights out anyway."""
    r25 = pd.DataFrame([{"code": "AAA", "constructor_id": "t", "classified": True,
                         "status": "Finished"}] * 10)
    r26 = pd.DataFrame(
        [{"Abbreviation": "AAA", "TeamName": "T", "ClassifiedPosition": "1",
          "Status": "Finished"}] * 8
        + [{"Abbreviation": "AAA", "TeamName": "T", "ClassifiedPosition": "R",
            "Status": "Did not start"}] * 4)
    rel = fit_reliability(r25, r26, warmup=150, samples=150, chains=1)
    row = rel.by_entry[rel.by_entry.driver == "AAA"].iloc[0]
    # Eight starts, zero in-race retirements — the four non-starts must not appear.
    assert row["starts"] == 8
    assert row["dnfs"] == 0
    assert row["p_dnf"] < 0.30
