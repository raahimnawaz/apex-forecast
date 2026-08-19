"""Tests for the championship projection.

The one that matters is `test_a_season_is_correlated_not_twelve_coin_flips`. Everything
else here is arithmetic; that test is the modelling claim.
"""

from __future__ import annotations

import numpy as np

from apex.championship import (
    RACE_POINTS,
    SPRINT_POINTS,
    _share_of_wins,
    project_championship,
)
from apex.reliability import simulate_race

ENTRIES = [("AAA", "Alpha"), ("BBB", "Alpha"), ("CCC", "Beta"), ("DDD", "Beta")]


def _theta(rows):
    return np.asarray(rows, dtype=float)


def test_ordering_agrees_with_simulate_race():
    """`_session` re-implements simulate_race's ordering; it must not drift from it.

    Compared in points rather than positions, because points are what the projection
    actually uses and they weight the places that matter.
    """
    n = 6
    entries = [(f"D{i}", f"T{i // 2}") for i in range(n)]
    theta = _theta([np.linspace(2.0, -1.0, n)])
    p_dnf = np.full(n, 0.12)

    probs = simulate_race(theta, p_dnf, sc_rate=0.3, n_sim=6000, likelihood="attrition")
    by_pos = np.zeros(n)
    by_pos[:min(len(RACE_POINTS), n)] = RACE_POINTS[:min(len(RACE_POINTS), n)]
    expected = probs @ by_pos

    proj = project_championship(theta, p_dnf, 0.3, entries, np.zeros(n),
                                [(1, False)], n_sim=30000)
    assert np.allclose(proj.exp_points, expected, atol=0.5)


def test_a_season_is_correlated_not_twelve_coin_flips():
    """One posterior draw must be held across the whole season.

    The posterior here says the field is one of two worlds: either AAA is dominant or
    CCC is. A correlated season commits to a world and runs away with it, so AAA's final
    points are bimodal and hugely spread. Redrawing the world every round would average
    the two out and give a narrow band around the middle — which is exactly the mistake
    that multiplying twelve marginal race distributions would make.
    """
    theta = _theta([[3.0, 0.0, -3.0, 0.0],      # world A: AAA dominant
                    [-3.0, 0.0, 3.0, 0.0]])     # world B: CCC dominant
    p_dnf = np.zeros(4)
    remaining = [(r, False) for r in range(12, 24)]

    proj = project_championship(theta, p_dnf, 0.0, ENTRIES, np.zeros(4),
                                remaining, n_sim=4000)

    aaa = proj.drivers.index("AAA")
    p10, p90 = proj.fan[0, -1, aaa], proj.fan[-1, -1, aaa]
    # Twelve races at 25 for winning and ~0 for losing: the two worlds sit ~250 apart.
    assert p90 - p10 > 150, f"final spread {p90 - p10:.0f} is too narrow to be correlated"

    # And the two dominant drivers split the title, because the posterior splits the worlds.
    assert proj.title_prob[aaa] > 0.35
    assert proj.title_prob[proj.drivers.index("CCC")] > 0.35
    assert proj.title_prob.sum() > 0.999


def test_a_sprint_round_awards_sprint_points_on_top_of_the_race():
    n = len(ENTRIES)
    theta = _theta([[1.0, 0.5, 0.0, -0.5]])
    p_dnf = np.zeros(n)

    plain = project_championship(theta, p_dnf, 0.0, ENTRIES, np.zeros(n),
                                 [(12, False)], n_sim=4000)
    sprint = project_championship(theta, p_dnf, 0.0, ENTRIES, np.zeros(n),
                                  [(12, True)], n_sim=4000)

    # Four cars, so only the top four of each table are ever awarded.
    assert np.isclose(plain.exp_points.sum(), RACE_POINTS[:n].sum())
    assert np.isclose(sprint.exp_points.sum(),
                      RACE_POINTS[:n].sum() + SPRINT_POINTS[:n].sum())


def test_points_already_scored_carry_through():
    theta = _theta([[0.0, 0.0, 0.0, 0.0]])
    now = np.array([100.0, 0.0, 0.0, 0.0])
    proj = project_championship(theta, np.zeros(4), 0.0, ENTRIES, now,
                                [(12, False)], n_sim=2000)

    assert np.allclose(proj.fan[:, 0, :], now)          # round 0 of the fan is today
    assert proj.exp_points[0] > 100
    # An equal field plus a 100-point head start over one race is not a contest.
    assert proj.title_prob[0] > 0.99


def test_constructor_points_are_the_sum_of_both_cars():
    theta = _theta([[1.0, 0.9, -1.0, -0.9]])
    now = np.array([10.0, 20.0, 5.0, 1.0])
    proj = project_championship(theta, np.zeros(4), 0.0, ENTRIES, now,
                                [(12, False)], n_sim=2000)

    assert proj.team_names == ["Alpha", "Beta"]
    assert np.allclose(proj.team_points_now, [30.0, 6.0])
    assert np.isclose(proj.team_exp_points.sum(), proj.exp_points.sum())
    assert np.isclose(proj.team_title_prob.sum(), 1.0)


def test_mathematically_eliminated_drivers_are_flagged():
    """Reported as impossible, not as a probability that rounds to zero."""
    now = np.array([500.0, 0.0, 0.0, 0.0])
    proj = project_championship(_theta([[0.0] * 4]), np.zeros(4), 0.0, ENTRIES, now,
                                [(12, False), (13, True)], n_sim=500)

    # One race and one sprint is at most 33 points; nobody on 0 can catch 500.
    assert proj.still_possible[0]
    assert not proj.still_possible[1:].any()


def test_ties_are_split_rather_than_awarded_to_the_lowest_index():
    final = np.array([[10.0, 10.0, 5.0],
                      [7.0, 3.0, 3.0]])
    assert np.allclose(_share_of_wins(final), [0.75, 0.25, 0.0])


def test_constructor_standings_can_be_supplied_when_a_driver_changed_teams():
    """BBB scored for Beta before moving to Alpha; those points must stay with Beta."""
    theta = _theta([[0.0] * 4])
    now = np.array([10.0, 40.0, 5.0, 5.0])          # BBB carries 40 points to Alpha
    standings = {"Alpha": 10.0, "Beta": 50.0}       # but Beta earned 40 of them

    proj = project_championship(theta, np.zeros(4), 0.0, ENTRIES, now, [(12, False)],
                                n_sim=500, team_points_now=standings)

    assert np.allclose(proj.team_points_now, [10.0, 50.0])
    assert np.isclose(proj.team_exp_points.sum(),
                      60.0 + RACE_POINTS[:4].sum())    # standings + one race of gains
