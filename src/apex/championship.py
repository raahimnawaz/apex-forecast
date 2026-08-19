"""Championship projection — title probability across the remaining rounds.

`PLAN.md` §3 view 5. This is the one view that asks a question no single race answers:
not "who wins on Sunday" but "who is champion in December", which depends on how the
remaining races correlate with each other.

**The correlation is the whole point, and it is why this cannot be built from
`simulate_race` output.** That function returns *marginal* position probabilities for one
race, averaged over the posterior. Multiplying twelve of those together would treat each
round as an independent draw and quietly assume the car that is quick in September is a
fresh coin flip in October. It is not: a fast car is fast every weekend, and a title race
is decided by exactly that persistence.

So a projected season here draws **one posterior sample and holds it for all twelve
rounds**, then draws fresh retirements, safety cars and race-day noise within each round.
Uncertainty about how good a car *is* stays correlated across the season; uncertainty
about what *happens* on a given Sunday does not. Averaging over seasons then gives a title
probability that carries both.

What this does not model, stated rather than hidden:

- **No grid.** Future qualifying has not happened, so the projection runs off the
  grid-free posterior. This is limitation 1 arriving in a second place: a championship
  projection cannot condition on a grid that does not exist yet.
- **No development.** Each car's strength is its posterior today, held flat to December.
  The random walk in `strength.py` is fitted on races that have run; extrapolating its
  drift forward would invent a development slope from eleven noisy steps.
- **No circuit effects.** Team form is circuit-dependent (limitation 10) and not
  forecastable, so every remaining round is drawn from the same season-level strength.
- **No penalties, no reliability trend, no team orders.**
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Both tables are *derived from the season on disk*, not transcribed from a rulebook:
# grouping the 11 races and 4 sprints of 2026 by finishing position gives these values
# with zero variance at every position, which also rules out a fastest-lap bonus.
RACE_POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1], dtype=float)
SPRINT_POINTS = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=float)

# A sprint is about a third of a race distance, and retirement risk turns out to scale
# with it: 5 retirements in 86 sprint starts against 41 in 235 race starts across 2026,
# a ratio of 0.333 against a distance share of 100/305 = 0.328 (Fisher p = 0.0069). So
# the race-fitted p_dnf is scaled rather than reused whole, and the scale is measured
# rather than assumed — using the race value would roughly triple sprint attrition.
SPRINT_DNF_SCALE = 0.333

FAN_QUANTILES = (10, 25, 50, 75, 90)


@dataclass
class Projection:
    """Everything the dashboard needs, and nothing that would have to be recomputed."""

    drivers: list[str]
    teams: list[str]
    rounds: list[int]                 # the remaining rounds, in order
    points_now: np.ndarray            # (n_entries,) points already scored
    title_prob: np.ndarray            # (n_entries,) P(driver is champion)
    exp_points: np.ndarray            # (n_entries,) expected final total
    fan: np.ndarray                   # (n_quantiles, n_rounds + 1, n_entries)
    team_names: list[str]             # unique, in projection order
    team_title_prob: np.ndarray       # (n_teams,) P(constructor title)
    team_points_now: np.ndarray       # (n_teams,)
    team_exp_points: np.ndarray       # (n_teams,)
    n_sim: int
    still_possible: np.ndarray        # (n_entries,) bool: can still reach the lead


def _session(rng: np.random.Generator, theta: np.ndarray, p_dnf: np.ndarray,
             sc_rate: float, table: np.ndarray, fcy_temperature: float,
             likelihood: str) -> np.ndarray:
    """Points scored by every entry in one session. Mirrors `simulate_race`'s ordering.

    theta is (n_sim, n): one row per projected season, already drawn from the posterior
    and held fixed across the season, so only the race-day noise is redrawn here.
    """
    n_sim, n = theta.shape
    retire = rng.random((n_sim, n)) < p_dnf[None, :]
    sc = rng.random(n_sim) < sc_rate
    scale = np.where(sc, fcy_temperature, 1.0)[:, None]
    g = rng.gumbel(size=(n_sim, n))

    key = theta / scale + g
    if likelihood == "attrition":
        key = -(-theta / scale + g)

    # Retirements sort behind every finisher; among themselves the order is random.
    key = np.where(retire, -1e9 + rng.random((n_sim, n)), key)
    order = np.argsort(-key, axis=1)

    by_position = np.zeros(n)
    by_position[:min(len(table), n)] = table[:min(len(table), n)]
    gained = np.empty((n_sim, n))
    np.put_along_axis(gained, order, np.broadcast_to(by_position, (n_sim, n)), axis=1)
    return gained


def project_championship(theta: np.ndarray, p_dnf: np.ndarray, sc_rate: float,
                         entries: list[tuple[str, str]], points_now: np.ndarray,
                         remaining: list[tuple[int, bool]], n_sim: int = 4000,
                         fcy_temperature: float = 1.9, likelihood: str = "attrition",
                         seed: int = 17,
                         team_points_now: dict[str, float] | None = None) -> Projection:
    """Project the title from here.

    `remaining` is [(round, is_sprint), ...] in order. `theta` is the grid-free posterior
    for `entries`, shape (n_posterior_samples, n_entries).

    `team_points_now` is passed separately rather than summed from `points_now` because
    24 drivers have held 22 seats this season: a driver who changed teams keeps their
    points, but their old team keeps the constructor points those results earned. Summing
    driver totals by current team would move them, and would silently credit the wrong
    constructor. Defaults to the sum when no standings are supplied.
    """
    rng = np.random.default_rng(seed)
    n_post, n = theta.shape
    if len(entries) != n:
        raise ValueError(f"{len(entries)} entries against theta with {n} columns")

    # One posterior draw per projected season, reused for every remaining round. This
    # single line is what makes the projection a season rather than twelve unrelated races.
    season_theta = theta[rng.integers(0, n_post, n_sim)]

    # Points still to be won are tracked on their own, so that a constructor total can be
    # built from this season's remaining gains plus the standings as they actually are.
    gains = np.zeros((n_sim, n))
    history = np.empty((len(remaining) + 1, n_sim, n))
    history[0] = points_now

    for i, (_rnd, is_sprint) in enumerate(remaining):
        if is_sprint:
            # The sprint runs first, and over a third of the distance (see the constant).
            gains += _session(rng, season_theta, p_dnf * SPRINT_DNF_SCALE, sc_rate,
                              SPRINT_POINTS, fcy_temperature, likelihood)
        gains += _session(rng, season_theta, p_dnf, sc_rate, RACE_POINTS,
                          fcy_temperature, likelihood)
        history[i + 1] = points_now + gains

    cum = points_now + gains
    title_prob = _share_of_wins(cum)

    drivers = [e[0] for e in entries]
    teams = [e[1] for e in entries]
    team_names = list(dict.fromkeys(teams))
    tix = np.array([team_names.index(t) for t in teams])
    team_gains = np.zeros((n_sim, len(team_names)))
    np.add.at(team_gains.T, tix, gains.T)
    if team_points_now is None:
        team_now = np.zeros(len(team_names))
        np.add.at(team_now, tix, points_now)
    else:
        team_now = np.array([float(team_points_now.get(t, 0.0)) for t in team_names])
    team_cum = team_now + team_gains

    # A driver is mathematically out once the maximum they can still score leaves them
    # behind the current leader's total. Cheaper and far more honest than reporting a
    # title probability of 0.0000 that is really "not seen in 4000 draws".
    max_gain = sum(RACE_POINTS[0] + (SPRINT_POINTS[0] if sp else 0.0)
                   for _, sp in remaining)
    still_possible = points_now + max_gain >= points_now.max()

    return Projection(
        drivers=drivers, teams=teams, rounds=[r for r, _ in remaining],
        points_now=points_now, title_prob=title_prob, exp_points=cum.mean(0),
        fan=np.percentile(history, FAN_QUANTILES, axis=1),
        team_names=team_names, team_title_prob=_share_of_wins(team_cum),
        team_points_now=team_now, team_exp_points=team_cum.mean(0),
        n_sim=n_sim, still_possible=still_possible,
    )


def _share_of_wins(final: np.ndarray) -> np.ndarray:
    """P(each column finishes top) with ties split evenly.

    F1 breaks a points tie on countback — most wins, then most seconds, and so on. That
    is not modelled: a tie is split. Ties are rare enough that this moves nothing, and
    inventing a countback would imply a precision the simulation does not have.
    """
    best = final.max(axis=1, keepdims=True)
    tied = final == best
    return (tied / tied.sum(axis=1, keepdims=True)).mean(axis=0)
