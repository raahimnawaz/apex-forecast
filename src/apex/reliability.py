"""Tier 1 — reliability, safety cars, and the machinery that makes a forecast unconditional.

Everything the rank model deliberately ignores lives here. Layer 1 answers "who is
quickest among the cars that finish"; this module answers "who finishes at all, and what
happens when the race is interrupted". Together they produce a forecast over the whole
field rather than over survivors.

Design follows Heilmeier et al. (2020), *Application of Monte Carlo Methods to Consider
Probabilistic Effects in a Race Simulation for Circuit Motorsport*, which treats
retirements and full-course-yellow phases as explicit stochastic processes drawn per
simulation run rather than as smoothed-in averages. The rates here are estimated from
this project's own 2025-2026 data instead of taken from the paper, because the 2026
regulations reset reliability along with everything else.

Two processes:

**Retirement.** A hierarchical logistic model with partial pooling across teams and
drivers. Pooling matters: Cadillac has retired 35% of the time across twenty starts, and
the raw rate is far too confident on that little evidence. Shrinking toward the grid mean
keeps a debut team's estimate honest without ignoring what it has shown.

**Full-course yellow.** Safety cars are the main route by which a midfield car wins — a
free pit stop, a compressed field, a restart. A model without them cannot produce that
outcome at all, which is exactly why an unmitigated Plackett-Luce forecast has an
implausibly thin tail. Rather than simulate the pit lane lap by lap, an FCY is modelled
as a temperature increase on the ordering: when one occurs, the finishing order is drawn
with more entropy, so track position is worth less and upsets become possible. That is a
reduced-form stand-in for the real mechanism, and it is labelled as one.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS


@dataclass
class Reliability:
    """Posterior mean P(retire) per driver-team pairing, plus the pieces behind it."""
    by_entry: pd.DataFrame        # driver, team, p_dnf, p_dnf_lo, p_dnf_hi, starts, dnfs
    grid_mean: float
    sc_rate: float                # P(at least one full safety car) per race
    vsc_rate: float               # P(at least one VSC) — reported, not simulated
    fcy_by_circuit: pd.DataFrame  # circuit, races, sc_rate (shrunk)


def _dnf_frame(results_2025: pd.DataFrame, results_2026: pd.DataFrame) -> pd.DataFrame:
    """One row per entry, with a retirement flag, across both seasons."""
    a = pd.DataFrame({
        "driver": results_2025["code"],
        "team": results_2025["constructor_id"],
        "dnf": (~results_2025["classified"]).astype(int),
        "season": 2025,
    })
    b = results_2026.copy()
    b = pd.DataFrame({
        "driver": b["Abbreviation"],
        "team": b["TeamName"],
        "dnf": (~b["ClassifiedPosition"].astype(str).str.isdigit()).astype(int),
        "season": 2026,
    })
    return pd.concat([a, b], ignore_index=True)


def _model(team_ix, driver_ix, dnf, n_teams, n_drivers):
    mu = numpyro.sample("mu", dist.Normal(-2.0, 1.5))          # grid-level log-odds
    sigma_team = numpyro.sample("sigma_team", dist.HalfNormal(1.0))
    sigma_driver = numpyro.sample("sigma_driver", dist.HalfNormal(0.5))

    team_raw = numpyro.sample("team_raw", dist.Normal(0, 1).expand([n_teams]).to_event(1))
    driver_raw = numpyro.sample("driver_raw", dist.Normal(0, 1).expand([n_drivers]).to_event(1))

    team = numpyro.deterministic("team", team_raw * sigma_team)
    driver = numpyro.deterministic("driver", driver_raw * sigma_driver)

    logit = mu + team[team_ix] + driver[driver_ix]
    numpyro.sample("obs", dist.Bernoulli(logits=logit), obs=dnf)


def fit_reliability(results_2025: pd.DataFrame, results_2026: pd.DataFrame,
                    laps_2026: pd.DataFrame | None = None,
                    warmup: int = 600, samples: int = 600, chains: int = 2,
                    seed: int = 20260726) -> Reliability:
    """Estimate retirement probability per entry and the FCY rate per race."""
    df = _dnf_frame(results_2025, results_2026)

    # Only the 2026 pairings matter for forecasting, but 2025 informs the team and
    # driver effects, so both seasons are fitted and 2026 entries are read back out.
    teams = sorted(df["team"].unique())
    drivers = sorted(df["driver"].unique())
    t_ix = np.array([teams.index(t) for t in df["team"]])
    d_ix = np.array([drivers.index(d) for d in df["driver"]])

    numpyro.set_host_device_count(chains)
    mcmc = MCMC(NUTS(_model, target_accept_prob=0.9), num_warmup=warmup,
                num_samples=samples, num_chains=chains, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), jnp.asarray(t_ix), jnp.asarray(d_ix),
             jnp.asarray(df["dnf"].to_numpy()), len(teams), len(drivers))
    post = mcmc.get_samples()

    mu = np.asarray(post["mu"])[:, None]
    team_eff = np.asarray(post["team"])
    driver_eff = np.asarray(post["driver"])

    cur = results_2026.copy()
    cur["dnf"] = (~cur["ClassifiedPosition"].astype(str).str.isdigit()).astype(int)
    pairs = (cur.groupby(["Abbreviation", "TeamName"])
                .agg(starts=("dnf", "size"), dnfs=("dnf", "sum")).reset_index())

    rows = []
    for r in pairs.itertuples():
        ti, di = teams.index(r.TeamName), drivers.index(r.Abbreviation)
        p = 1.0 / (1.0 + np.exp(-(mu[:, 0] + team_eff[:, ti] + driver_eff[:, di])))
        rows.append({
            "driver": r.Abbreviation, "team": r.TeamName,
            "p_dnf": float(p.mean()),
            "p_dnf_lo": float(np.percentile(p, 5.5)),
            "p_dnf_hi": float(np.percentile(p, 94.5)),
            "starts": int(r.starts), "dnfs": int(r.dnfs),
            "raw_rate": float(r.dnfs / max(r.starts, 1)),
        })
    by_entry = pd.DataFrame(rows).sort_values("p_dnf", ascending=False).reset_index(drop=True)

    # --- safety car rate ---------------------------------------------------------------
    # Only a *full* safety car is simulated as a disruption. Every 2026 race so far has
    # seen either a safety car or a VSC, so "any full-course yellow" fires 100% of the
    # time and is therefore not a distinguishing event at all — applying a disruption on
    # every run would just be a constant temperature rise dressed up as a process. A VSC
    # also freezes the field rather than bunching it, so it shuffles far less. Six of
    # eleven races had a genuine safety car; that is the number worth simulating.
    sc_rate = vsc_rate = 0.0
    fcy_circuit = pd.DataFrame()
    if laps_2026 is not None and not laps_2026.empty:
        race = laps_2026[laps_2026["session"] == "R"]
        per_race = race.groupby(["round", "event"]).agg(
            sc=("has_sc", "any"), vsc=("has_vsc", "any")).reset_index()
        sc_rate = float(per_race["sc"].mean())
        vsc_rate = float(per_race["vsc"].mean())
        # One race per circuit is far too little to trust a per-circuit rate, so each is
        # shrunk hard toward the season mean with a pseudo-count.
        k = 4.0
        fcy_circuit = per_race.groupby("event", as_index=False).agg(
            races=("sc", "size"), hits=("sc", "sum"))
        fcy_circuit["sc_rate"] = ((fcy_circuit["hits"] + k * sc_rate)
                                  / (fcy_circuit["races"] + k))

    return Reliability(by_entry=by_entry, grid_mean=float(df["dnf"].mean()),
                       sc_rate=sc_rate, vsc_rate=vsc_rate, fcy_by_circuit=fcy_circuit)


def simulate_race(theta: np.ndarray, p_dnf: np.ndarray, sc_rate: float,
                  n_sim: int = 600, fcy_temperature: float = 1.9,
                  likelihood: str = "attrition", seed: int = 11) -> np.ndarray:
    """Full-field finishing distribution, retirements and safety cars included.

    Each run: draw who retires, order the survivors, then stack the retirements behind
    them in reverse draw order (a car that stops early is classified below one that stops
    late). With probability `fcy_rate` the run is an interrupted race and the ordering is
    drawn at a higher temperature, which is what finally gives a midfield car a route to
    the front.

    Returns (n_entries, n_entries) position probabilities that sum to 1 per driver.
    """
    rng = np.random.default_rng(seed)
    S, n = theta.shape
    counts = np.zeros((n, n))

    for _ in range(n_sim):
        retire = rng.random((S, n)) < p_dnf[None, :]
        sc = rng.random(S) < sc_rate
        # Temperature acts on the spread of theta: hotter means less deterministic.
        scale = np.where(sc, fcy_temperature, 1.0)[:, None]
        g = rng.gumbel(size=(S, n))
        key = theta / scale + g
        if likelihood == "attrition":
            key = -(-theta / scale + g)

        # Retirements sort behind every finisher; among themselves the order is random,
        # which is honest — we are not modelling *when* each car stops.
        key = np.where(retire, -1e9 + rng.random((S, n)), key)
        order = np.argsort(-key, axis=1)
        for pos in range(n):
            np.add.at(counts, (order[:, pos], pos), 1)

    return counts / counts.sum(axis=1, keepdims=True)
