"""Recovery tests for Layer 0.

The point of a deconvolution model is that it returns the effects you put in. These
tests synthesise laps with *known* driver offsets, degradation slopes and a lap trend,
then check the fit recovers them. If a refactor breaks the design matrix, these fail
loudly rather than quietly shifting everyone's pace by a tenth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apex.pace import add_gap_ahead, fit_event, prepare_race_laps

RNG = np.random.default_rng(20260724)

TRUE_DRIVER = {  # seconds vs field mean
    "AAA": -0.60, "BBB": -0.45, "CCC": -0.30, "DDD": -0.10, "EEE": -0.05,
    "FFF": 0.05, "GGG": 0.15, "HHH": 0.30, "III": 0.45, "JJJ": 0.55,
}
TRUE_DEG = {"SOFT": 0.09, "MEDIUM": 0.05, "HARD": 0.03}  # s per lap of tyre age
TRUE_TREND = -0.045  # s per lap, fuel burn + track evolution combined
BASE = 90.0


def stint_plan(n_laps: int, offset: int) -> list[tuple[str, int, int]]:
    """Three stints with a driver-specific pit window.

    The offset matters, and not cosmetically. If every driver pitted on the same lap,
    tyre age would be an exact affine function of lap number within each compound, and
    the lap trend would be perfectly collinear with the degradation slopes — only their
    sum would be identifiable. Real races identify the two separately *because* drivers
    stop at different laps. The fixture has to reproduce that to be a fair test.
    """
    a = max(6, min(n_laps - 2, 16 + offset))
    b = max(a + 3, min(n_laps - 1, 35 + offset))
    plan = [("SOFT", 1, a), ("MEDIUM", a + 1, b), ("HARD", b + 1, n_laps)]
    return [(c, s, e) for c, s, e in plan if s <= e]


def synth_race(n_laps: int = 55, noise_s: float = 0.12) -> pd.DataFrame:
    rows = []
    for i, (drv, eff) in enumerate(TRUE_DRIVER.items()):
        for compound, start, end in stint_plan(n_laps, offset=(i % 7) - 3):
            for lap in range(start, end + 1):
                age = lap - start + 1
                t = (BASE + eff + TRUE_DEG[compound] * age + TRUE_TREND * (lap - n_laps / 2)
                     + RNG.normal(0, noise_s))
                rows.append({
                    "Driver": drv, "Team": f"T{drv[0]}", "LapNumber": lap,
                    "lap_time_s": t, "Compound": compound, "TyreLife": age,
                    "IsAccurate": True, "is_green": True, "is_pit_in": False,
                    "is_pit_out": False, "lap_end_s": lap * BASE + eff * lap,
                    "season": 2026, "round": 99, "session": "R", "event": "Synthetic GP",
                })
    return pd.DataFrame(rows)


def test_prepare_drops_non_representative_laps():
    df = synth_race()
    df.loc[df["LapNumber"] <= 2, "LapNumber"] = 1        # start-phase laps
    df.loc[df.index[:5], "is_pit_in"] = True
    df.loc[df.index[5:10], "is_green"] = False
    out = prepare_race_laps(df, 2026, 99)
    assert not out["is_pit_in"].any()
    assert out["is_green"].all()
    assert (out["LapNumber"] > 2).all()


def test_gap_ahead_is_within_lap_and_leader_is_clear_air():
    df = synth_race()
    g = add_gap_ahead(df)
    lap5 = g[g["LapNumber"] == 5].sort_values("lap_end_s")
    assert np.isinf(lap5["gap_ahead_s"].iloc[0])          # leader has no car ahead
    assert lap5["dirty_air"].iloc[0] == 0.0
    assert (g["dirty_air"] >= 0).all()


def test_recovers_driver_effects():
    ep = fit_event(synth_race(), 2026, 99)
    assert ep is not None
    got = dict(zip(ep.drivers["Driver"], ep.drivers["pace_s"]))
    centred = {k: v - np.mean(list(TRUE_DRIVER.values())) for k, v in TRUE_DRIVER.items()}
    for drv, want in centred.items():
        assert got[drv] == pytest.approx(want, abs=0.05), f"{drv}: {got[drv]:.3f} vs {want:.3f}"


def test_recovers_degradation_slopes():
    ep = fit_event(synth_race(), 2026, 99)
    got = dict(zip(ep.deg["compound"], ep.deg["deg_s_per_lap"]))
    for compound, want in TRUE_DEG.items():
        if compound in got:
            assert got[compound] == pytest.approx(want, abs=0.01), f"{compound}: {got[compound]:.4f}"


def test_recovers_lap_trend():
    ep = fit_event(synth_race(), 2026, 99)
    assert ep.lap_trend_s == pytest.approx(TRUE_TREND, abs=0.01)


def test_driver_effects_are_centred_on_the_field():
    ep = fit_event(synth_race(), 2026, 99)
    assert ep.drivers["pace_s"].mean() == pytest.approx(0.0, abs=1e-9)


def test_reference_driver_se_is_imputed_not_nan():
    """The treatment-coded reference level has no SE of its own; it must not leak NaN
    into the JSON payload."""
    ep = fit_event(synth_race(), 2026, 99)
    assert ep.drivers["se_s"].notna().all()
    assert ep.drivers["se_imputed"].sum() == 1


def test_too_little_data_returns_none():
    tiny = synth_race(n_laps=6)
    assert fit_event(tiny, 2026, 99) is None
