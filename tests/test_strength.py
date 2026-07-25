"""Tests for Layer 1.

The Plackett-Luce likelihood is checked against its closed form rather than against
itself: for a small field the probability of an ordering can be written out by hand,
so a sign error or an off-by-one in the denominator has nowhere to hide.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from apex.strength import (
    CONSTRUCTOR_CONTINUITY,
    _plackett_luce_loglik,
    build,
    predict_order,
)


def test_loglik_matches_closed_form():
    """P(a>b>c) = e^a/(e^a+e^b+e^c) · e^b/(e^b+e^c) · 1."""
    theta = np.array([[1.5, 0.3, -0.7]])
    valid = np.array([[True, True, True]])
    e = np.exp(theta[0])
    expected = np.log(e[0] / e.sum()) + np.log(e[1] / (e[1] + e[2]))
    got = float(_plackett_luce_loglik(jnp.asarray(theta), jnp.asarray(valid)))
    assert got == pytest.approx(expected, abs=1e-5)


def test_padding_is_ignored():
    """Padded slots must not enter the numerator or any denominator."""
    theta = np.array([[1.5, 0.3, -0.7, 99.0]])
    valid = np.array([[True, True, True, False]])
    e = np.exp(theta[0][:3])
    expected = np.log(e[0] / e.sum()) + np.log(e[1] / (e[1] + e[2]))
    got = float(_plackett_luce_loglik(jnp.asarray(theta), jnp.asarray(valid)))
    assert got == pytest.approx(expected, abs=1e-5)


def test_truncation_drops_tail_terms_only():
    """Truncating to the top M keeps M numerators but leaves denominators intact."""
    theta = np.array([[2.0, 1.0, 0.0, -1.0]])
    valid = np.ones((1, 4), dtype=bool)
    e = np.exp(theta[0])
    top2 = np.log(e[0] / e.sum()) + np.log(e[1] / e[1:].sum())
    got = float(_plackett_luce_loglik(jnp.asarray(theta), jnp.asarray(valid), top_m=2))
    assert got == pytest.approx(top2, abs=1e-5)
    # The tail still contributes when untruncated, so the two must differ.
    full = float(_plackett_luce_loglik(jnp.asarray(theta), jnp.asarray(valid)))
    assert full < top2


def test_better_ordering_is_more_likely():
    theta = np.array([[2.0, 1.0, 0.0]])
    valid = np.ones((1, 3), dtype=bool)
    correct = float(_plackett_luce_loglik(jnp.asarray(theta), jnp.asarray(valid)))
    reversed_ = float(_plackett_luce_loglik(jnp.asarray(theta[:, ::-1]), jnp.asarray(valid)))
    assert correct > reversed_


def _fake_seasons():
    codes = ["AAA", "BBB", "CCC", "DDD"]
    teams25 = ["mclaren", "mclaren", "ferrari", "ferrari"]
    rows25 = []
    for rnd in range(1, 6):
        for pos, (c, t) in enumerate(zip(codes, teams25), start=1):
            rows25.append({"season": 2025, "round": rnd, "event": f"R{rnd}", "code": c,
                           "constructor_id": t, "position": pos, "classified": True})
    r25 = pd.DataFrame(rows25)

    rows26 = []
    for rnd in range(1, 4):
        for pos, c in enumerate(codes, start=1):
            rows26.append({"round": rnd, "event": f"E{rnd}", "Abbreviation": c,
                           "TeamName": "McLaren" if c in ("AAA", "BBB") else "Ferrari",
                           "Position": pos, "ClassifiedPosition": str(pos)})
    return r25, pd.DataFrame(rows26)


def test_build_shapes_and_era_flags():
    r25, r26 = _fake_seasons()
    d = build(r25, r26)
    assert len(d.races) == 8
    assert (d.era == 0).sum() == 5 and (d.era == 1).sum() == 3
    assert d.n_2026_rounds == 3
    assert d.valid.all()
    # Round index is 0-based within 2026 and parked at 0 for 2025.
    assert d.round_ix[d.era == 1].tolist() == [0, 1, 2]
    assert set(d.round_ix[d.era == 0].tolist()) == {0}


def test_winner_is_slot_zero():
    r25, r26 = _fake_seasons()
    d = build(r25, r26)
    first_2026 = d.races[d.races["season"] == 2026].iloc[0]
    assert first_2026["codes"][0] == "AAA"


def test_unmapped_constructor_raises():
    r25, r26 = _fake_seasons()
    r25.loc[0, "constructor_id"] = "not_a_team"
    with pytest.raises(ValueError, match="unmapped"):
        build(r25, r26)


def test_continuity_map_covers_the_2025_grid():
    """Every 2025 entrant must resolve to a 2026 entity, or build() will reject the season."""
    assert CONSTRUCTOR_CONTINUITY["sauber"] == "Audi"
    assert CONSTRUCTOR_CONTINUITY["rb"] == "Racing Bulls"
    assert len(set(CONSTRUCTOR_CONTINUITY.values())) == len(CONSTRUCTOR_CONTINUITY)


def test_predict_order_ranks_by_theta():
    """A stronger driver-car pairing must draw a higher win probability."""
    r25, r26 = _fake_seasons()
    d = build(r25, r26)
    S = 200
    n_c, n_t = len(d.constructors), d.n_2026_rounds
    car = np.zeros((S, n_c, n_t))
    car[:, d.constructors.index("McLaren"), :] = 1.5
    post = {
        "skill": np.tile(np.array([1.0, 0.5, 0.0, -0.5]), (S, 1)),
        "car26": car,
        "sigma_walk": np.full(S, 0.01),
        "sigma_skill": np.full(S, 1.0),
    }
    entries = [("AAA", "McLaren"), ("BBB", "McLaren"), ("CCC", "Ferrari"), ("DDD", "Ferrari")]
    probs, theta = predict_order(post, d, entries, n_sim=60, seed=1)

    assert probs.shape == (4, 4)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)
    p_win = probs[:, 0]
    assert p_win[0] > p_win[1] > p_win[2] > p_win[3]
    assert theta[:, 0].mean() > theta[:, 2].mean()
