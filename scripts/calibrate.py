"""Layer 3 — walk-forward calibration.

Answers the only question that matters about a forecast: is it any good?

For each evaluation round k, the model is refitted on 2025 plus 2026 rounds 1..k-1 and
asked to predict round k, which it has never seen. Nothing about round k touches the fit,
so this is genuine out-of-sample scoring rather than a fit statistic.

Every candidate — including the naive baselines — is expressed as the same kind of object,
a Plackett-Luce distribution over finishing orders, so they can be scored with the same
metrics. A baseline is not a point prediction here; it is a probabilistic forecast whose
one scale parameter is fitted on the training races by maximum likelihood. That makes
"grid position predicts the finish" a genuinely strong opponent rather than a straw man.

Metrics
-------
- **RPS** (ranked probability score): the right metric for ordered outcomes. Lower is
  better. It rewards putting mass *near* the truth, unlike log-loss which only cares
  about the exact cell.
- **log-loss on win / podium / points**: calibration of the headline probabilities.
- **Spearman**: does the predicted order match the actual order at all.
"""

from __future__ import annotations

import argparse
import glob
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from apex.history import fetch_season_results
from apex.paths import PROCESSED, RAW, REPORTS
from apex.scoring import score
from apex.strategy import (
    fit_team_degradation,
    season_average_profile,
    seconds_to_theta,
    theta_offsets,
    with_circuit_profile,
)
from apex.strength import TOP_M, build, fit, grid_advantage, predict_order

warnings.filterwarnings("ignore")

ALL_VARIANTS = [
    ("model: forward",           False, "forward"),
    ("model: forward+grid",      True,  "forward"),
    ("model: attrition+grid",    True,  "attrition"),
    ("model: contaminated+grid", True,  "contaminated"),
]

# Layer 2a is not a separate fit. It is the shipped variant plus a per-entry theta offset,
# so it is scored against the very posterior it modifies — a paired comparison in which the
# only difference is the strategy term.
STRATEGY_BASE = "model: attrition+grid"
STRATEGY_NAME = "model: attrition+grid+strategy"


# --------------------------------------------------------------------------------------
# Plackett-Luce helpers for the baselines
# --------------------------------------------------------------------------------------

def pl_loglik(theta_by_race: list[np.ndarray], top_m: int | None = TOP_M) -> float:
    """Log-likelihood of the observed orders; each array is already sorted by finish."""
    total = 0.0
    for th in theta_by_race:
        k = len(th) if top_m is None else min(top_m, len(th))
        for i in range(k):
            tail = th[i:]
            total += th[i] - (np.max(tail) + np.log(np.sum(np.exp(tail - np.max(tail)))))
    return total


def fit_scale(feats_by_race: list[np.ndarray]) -> float:
    """One-parameter MLE for a baseline: theta = c * feature."""
    def nll(c):
        return -pl_loglik([c * f for f in feats_by_race])
    res = minimize_scalar(nll, bounds=(0.0, 20.0), method="bounded")
    return float(res.x)


def pl_position_probs(theta: np.ndarray, n_sim: int = 6000, seed: int = 11) -> np.ndarray:
    """Position distribution by Gumbel-max sampling — exact draws from Plackett-Luce."""
    rng = np.random.default_rng(seed)
    n = len(theta)
    counts = np.zeros((n, n))
    g = rng.gumbel(size=(n_sim, n))
    order = np.argsort(-(theta[None, :] + g), axis=1)
    for p in range(n):
        np.add.at(counts, (order[:, p], p), 1)
    return counts / counts.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------------------
# data assembly
# --------------------------------------------------------------------------------------

def race_frame(r26: pd.DataFrame, rnd: int) -> pd.DataFrame:
    g = r26[(r26["round"] == rnd) & r26["ClassifiedPosition"].astype(str).str.isdigit()].copy()
    g["Position"] = pd.to_numeric(g["Position"])
    g["GridPosition"] = pd.to_numeric(g["GridPosition"])
    return g.sort_values("Position").reset_index(drop=True)


def standings_before(r26: pd.DataFrame, rnd: int) -> dict[str, int]:
    prior = r26[r26["round"] < rnd]
    pts = prior.groupby("Abbreviation")["Points"].sum().sort_values(ascending=False)
    return {code: i + 1 for i, code in enumerate(pts.index)}


def last_race_order(r26: pd.DataFrame, rnd: int) -> dict[str, int]:
    prev = race_frame(r26, rnd - 1)
    return {r.Abbreviation: int(r.Position) for r in prev.itertuples()}


# --------------------------------------------------------------------------------------

def strategy_theta(laps: pd.DataFrame, r26: pd.DataFrame, rnd: int, event: str,
                   teams: list[str], pace: pd.DataFrame,
                   strength: pd.DataFrame) -> np.ndarray | None:
    """Layer 2a offsets for round `rnd`, built without touching round `rnd`.

    Everything here is restricted to rounds strictly before the target: the per-team
    degradation pooling, the pit-loss estimates and the reference circuits the offset is
    measured against. The target circuit itself gets a season-average tyre profile, since
    its own race is the thing being forecast — see `strategy.season_average_profile` for
    why Friday practice cannot stand in for it.
    """
    prior_laps = laps[laps["round"] < rnd]
    if prior_laps.empty:
        return None
    tyres = fit_team_degradation(prior_laps, 2026)
    if tyres.by_circuit.empty or tyres.by_team.empty:
        return None

    # Race distance is published before the weekend, so using it is not leakage. It is
    # read from the target round's laps only because that is where this repo stores it.
    tgt = laps[laps["round"] == rnd]
    n_laps = int(tgt["LapNumber"].max()) if not tgt.empty else None
    tyres = with_circuit_profile(tyres, season_average_profile(tyres, event, rnd),
                                 n_laps=n_laps)

    scale = seconds_to_theta(pace[pace["round"] < rnd], strength)
    if scale <= 0:
        return None
    ref = [e for e in tyres.race_laps["event"].unique() if e != event]
    if not ref:
        return None
    return theta_offsets(tyres, event, teams, scale, training_events=ref)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-round", type=int, default=5)
    ap.add_argument("--to-round", type=int, default=None,
                    help="last round to evaluate. Defaults to the newest round with "
                         "race results on disk, so the sample grows as the season "
                         "runs instead of staying frozen at whatever was current "
                         "when this default was written.")
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--variants", type=str, default="all",
                    help="comma-separated variant names, or 'all'. A focused A/B is far "
                         "cheaper than the full bake-off when testing one new layer.")
    args = ap.parse_args()

    r25 = fetch_season_results(2025)
    files = sorted(glob.glob(str(RAW / "results_2026_R*_R.parquet")))
    r26 = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    sfiles = sorted(glob.glob(str(RAW / "results_2026_R*_S.parquet")))
    sprints_all = (pd.concat([pd.read_parquet(f) for f in sfiles], ignore_index=True)
                   if sfiles else pd.DataFrame())

    # A hardcoded end round silently stops the sample growing: every later race is
    # ingested, scored and published, but never enters the backtest that decides whether
    # the model is any good. The published 7-race summary was produced by passing
    # --to-round by hand, which a plain `make calibrate` then could not reproduce.
    if args.to_round is None:
        args.to_round = int(r26["round"].max())
        print(f"evaluating through round {args.to_round} (newest with results on disk)")

    wanted = {x.strip() for x in args.variants.split(",")} if args.variants != "all" else None

    def want(name: str) -> bool:
        return wanted is None or name in wanted or name.replace("model: ", "") in wanted

    strategy = want(STRATEGY_NAME)
    laps26 = pace26 = None
    if strategy:
        lp = PROCESSED / "laps_2026.parquet"
        pp = PROCESSED / "pace_2026.parquet"
        if lp.exists() and pp.exists():
            laps26 = pd.read_parquet(lp)
            laps26 = laps26[(laps26["season"] == 2026) & (laps26["session"] == "R")]
            pace26 = pd.read_parquet(pp)
        else:
            print("no laps/pace tables — skipping the strategy variant")
            strategy = False

    rows = []
    for rnd in range(args.from_round, args.to_round + 1):
        actual = race_frame(r26, rnd)
        if actual.empty:
            continue
        codes = actual["Abbreviation"].tolist()
        teams = actual["TeamName"].tolist()
        grid = actual["GridPosition"].to_numpy(dtype=float)
        pos = actual["Position"].to_numpy(dtype=float)
        n = len(codes)
        print(f"\n=== round {rnd}: {actual['event'].iloc[0]} ({n} classified) ===")

        train26 = r26[r26["round"] < rnd]
        # Sprints from earlier rounds only — a walk-forward that leaks the target
        # weekend's own sprint would not be out of sample.
        train_sp = (sprints_all[sprints_all["round"] < rnd]
                    if not sprints_all.empty else pd.DataFrame())
        d = build(r25, train26, sprints_2026=train_sp)

        # ---- baselines, each a fitted one-parameter Plackett-Luce --------------------
        train_feats_grid, train_feats_stand, train_feats_last = [], [], []
        for tr in sorted(train26["round"].unique()):
            g = race_frame(r26, int(tr))
            if g.empty:
                continue
            a = grid_advantage(g["GridPosition"].to_numpy(dtype=float))
            train_feats_grid.append(a - a.mean())
            st = standings_before(r26, int(tr))
            if st:
                v = grid_advantage(np.array([st.get(c, len(st) + 1) for c in g["Abbreviation"]]))
                train_feats_stand.append(v - v.mean())
            if int(tr) > 1:
                lr = last_race_order(r26, int(tr))
                v = grid_advantage(np.array([lr.get(c, n + 1) for c in g["Abbreviation"]]))
                train_feats_last.append(v - v.mean())

        preds = {}

        a = grid_advantage(grid); a = a - a.mean()
        preds["baseline: grid"] = pl_position_probs(fit_scale(train_feats_grid) * a)

        st = standings_before(r26, rnd)
        v = grid_advantage(np.array([st.get(c, len(st) + 1) for c in codes])); v = v - v.mean()
        preds["baseline: standings"] = pl_position_probs(fit_scale(train_feats_stand) * v)

        lr = last_race_order(r26, rnd)
        v = grid_advantage(np.array([lr.get(c, n + 1) for c in codes])); v = v - v.mean()
        preds["baseline: last race"] = pl_position_probs(fit_scale(train_feats_last) * v)

        # ---- model variants ------------------------------------------------------------
        # Three rank likelihoods, each a published answer to the same question, tested
        # head to head rather than chosen by argument.
        entries = list(zip(codes, teams))
        # The strategy variant has no fit of its own, so it drags its base variant in.
        variants = [v for v in ALL_VARIANTS if want(v[0]) or (strategy and v[0] == STRATEGY_BASE)]
        for name, use_grid, lik in variants:
            mcmc = fit(d, warmup=args.warmup, samples=args.samples, chains=args.chains,
                       use_grid=use_grid, likelihood=lik)
            post = mcmc.get_samples()
            event = str(actual["event"].iloc[0])

            # Layer 2a rides on top of an already-fitted variant rather than being its own
            # model, so it is applied to the attrition+grid posterior and scored against
            # that same posterior without it. That makes the comparison paired: any
            # difference is the strategy offsets and nothing else.
            offs = None
            if strategy and name == STRATEGY_BASE:
                car = np.asarray(post["car26"])[:, :, -1]
                strength_now = pd.DataFrame({"constructor": d.constructors,
                                             "car_2026_latest": car.mean(0)})
                offs = strategy_theta(laps26, r26, rnd, event, teams, pace26, strength_now)

            p, _ = predict_order(post, d, entries, n_sim=300, likelihood=lik,
                                 grid=grid.tolist() if use_grid else None,
                                 circuit=event)
            preds[name] = p
            if offs is not None:
                p2, _ = predict_order(post, d, entries, n_sim=300, likelihood=lik,
                                      grid=grid.tolist() if use_grid else None,
                                      circuit=event, theta_offset=offs)
                preds[STRATEGY_NAME] = p2
                print(f"    strategy offsets: spread {offs.max() - offs.min():.3f} theta, "
                      f"max |offset| {np.abs(offs).max():.3f}")
            if lik == "contaminated":
                print(f"    eps (share of positions decided by chaos) = "
                      f"{float(np.mean(post['eps'])):.3f}")

        for name, p in preds.items():
            s = score(p, pos)
            s.update({"round": rnd, "model": name, "n": n})
            rows.append(s)
            print(f"  {name:<24} RPS {s['rps']:.4f}  ll_win {s['ll_win']:.4f}  "
                  f"ll_podium {s['ll_podium']:.4f}  rho {s['spearman']:+.3f}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("nothing evaluated")
        return 1

    out = df.groupby("model").agg(
        rps=("rps", "mean"), ll_win=("ll_win", "mean"), ll_podium=("ll_podium", "mean"),
        ll_points=("ll_points", "mean"), spearman=("spearman", "mean"),
        races=("round", "nunique")).sort_values("rps")

    print("\n" + "=" * 78)
    print(f"WALK-FORWARD SUMMARY — rounds {args.from_round}-{args.to_round}, "
          f"out of sample (lower RPS/log-loss better)")
    print("=" * 78)
    print(out.round(4).to_string())

    best = out.index[0]
    print(f"\nbest by RPS: {best}")
    grid_bl = out.loc["baseline: grid", "rps"] if "baseline: grid" in out.index else None
    if grid_bl is not None:
        for m in out.index:
            if m.startswith("model:"):
                d_rps = out.loc[m, "rps"] - grid_bl
                verdict = "beats" if d_rps < 0 else "LOSES TO"
                print(f"  {m} {verdict} the grid baseline by {abs(d_rps):.4f} RPS")

    df.to_csv(REPORTS / "calibration_walkforward.csv", index=False)
    out.to_csv(REPORTS / "calibration_summary.csv")
    out.reset_index().to_parquet(PROCESSED / "calibration_summary.parquet", index=False)
    print(f"\nwrote {REPORTS / 'calibration_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
