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
from scipy.stats import spearmanr

from apex.history import fetch_season_results
from apex.paths import PROCESSED, RAW, REPORTS
from apex.strength import TOP_M, build, fit, grid_advantage, predict_order

warnings.filterwarnings("ignore")
EPS = 1e-12


# --------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------

def rps(probs: np.ndarray, actual_ix: np.ndarray) -> float:
    """Mean ranked probability score over the drivers in one race.

    probs[i, p] = P(driver i finishes in position p). actual_ix[i] = true position index.
    """
    n = probs.shape[1]
    out = []
    for i, a in enumerate(actual_ix):
        cdf = np.cumsum(probs[i])
        step = (np.arange(n) >= a).astype(float)
        out.append(np.sum((cdf - step) ** 2) / (n - 1))
    return float(np.mean(out))


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def score(probs: np.ndarray, actual_pos: np.ndarray) -> dict:
    n = probs.shape[0]
    ix = actual_pos.astype(int) - 1
    exp_pos = (probs * np.arange(1, n + 1)).sum(1)
    return {
        "rps": rps(probs, ix),
        "ll_win": logloss(probs[:, 0], (ix == 0).astype(float)),
        "ll_podium": logloss(probs[:, :3].sum(1), (ix < 3).astype(float)),
        "ll_points": logloss(probs[:, :10].sum(1), (ix < 10).astype(float)),
        "spearman": float(spearmanr(exp_pos, actual_pos).statistic),
    }


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

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-round", type=int, default=5)
    ap.add_argument("--to-round", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--chains", type=int, default=2)
    args = ap.parse_args()

    r25 = fetch_season_results(2025)
    files = sorted(glob.glob(str(RAW / "results_2026_R*_R.parquet")))
    r26 = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    sfiles = sorted(glob.glob(str(RAW / "results_2026_R*_S.parquet")))
    sprints_all = (pd.concat([pd.read_parquet(f) for f in sfiles], ignore_index=True)
                   if sfiles else pd.DataFrame())

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
        variants = [
            ("model: forward",           False, "forward"),
            ("model: forward+grid",      True,  "forward"),
            ("model: attrition+grid",    True,  "attrition"),
            ("model: contaminated+grid", True,  "contaminated"),
        ]
        for name, use_grid, lik in variants:
            mcmc = fit(d, warmup=args.warmup, samples=args.samples, chains=args.chains,
                       use_grid=use_grid, likelihood=lik)
            post = mcmc.get_samples()
            p, _ = predict_order(post, d, entries, n_sim=300, likelihood=lik,
                                 grid=grid.tolist() if use_grid else None,
                                 circuit=str(actual["event"].iloc[0]))
            preds[name] = p
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
