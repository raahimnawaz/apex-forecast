"""Score a published forecast against the race that actually happened.

This is the only test that cannot be gamed: the forecast JSON was written and committed
before the race existed. Nothing here refits anything — it reads what was published and
compares it to the result.

Scored two ways on purpose:
  classified — against finishers only, which is what the model claims to predict
  full       — against the whole field including retirements, which is how a reader
               would actually use the numbers

The gap between those two is the cost of having no reliability model.
"""

from __future__ import annotations

import argparse
import json

import fastf1
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from apex.paths import CACHE, REPORTS, WEB_DATA
from apex.strength import grid_advantage

fastf1.Cache.enable_cache(str(CACHE))
EPS = 1e-12


def rps(probs: np.ndarray, actual_ix: np.ndarray) -> float:
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
    n = probs.shape[1]
    ix = np.clip(actual_pos.astype(int) - 1, 0, n - 1)
    exp_pos = (probs * np.arange(1, n + 1)).sum(1)
    return {
        "rps": round(rps(probs, ix), 4),
        "ll_win": round(logloss(probs[:, 0], (ix == 0).astype(float)), 4),
        "ll_podium": round(logloss(probs[:, :3].sum(1), (ix < 3).astype(float)), 4),
        "ll_points": round(logloss(probs[:, :10].sum(1), (ix < 10).astype(float)), 4),
        "spearman": round(float(spearmanr(exp_pos, actual_pos).statistic), 4),
    }


def pl_probs(theta: np.ndarray, n_sim: int = 20000, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(theta)
    counts = np.zeros((n, n))
    order = np.argsort(-(theta[None, :] + rng.gumbel(size=(n_sim, n))), axis=1)
    for p in range(n):
        np.add.at(counts, (order[:, p], p), 1)
    return counts / counts.sum(axis=1, keepdims=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--round", type=int, required=True)
    args = ap.parse_args()

    published = json.loads((WEB_DATA / f"strength_{args.season}.json").read_text())
    fc = pd.DataFrame(published["forecast"])
    matrix = published["position_matrix"]
    drivers = matrix["drivers"]
    probs_all = np.asarray(matrix["probs"])

    ses = fastf1.get_session(args.season, args.round, "R")
    ses.load(laps=False, telemetry=False, weather=False, messages=False)
    res = ses.results.copy()
    res["Position"] = pd.to_numeric(res["Position"])
    res["GridPosition"] = pd.to_numeric(res["GridPosition"])
    res["classified"] = res["ClassifiedPosition"].astype(str).str.isdigit()

    print(f"=== {ses.event['EventName']} — forecast published "
          f"{published['generated_utc']} ===\n")

    # ---- the grid the model assumed vs the grid that actually formed -------------------
    q = fc.set_index("driver")["grid"].to_dict()
    actual_grid = res.set_index("Abbreviation")["GridPosition"].to_dict()
    moved = [(d, int(q[d]), int(actual_grid[d]))
             for d in q if d in actual_grid and int(q[d]) != int(actual_grid[d])]
    if moved:
        print("GRID PENALTIES the forecast did not know about:")
        for d, assumed, real in sorted(moved, key=lambda x: x[2]):
            print(f"  {d}: forecast assumed P{assumed}, actually started P{real} "
                  f"({real - assumed:+d})")
        print()

    rows = []
    for label, subset in (("classified", res[res["classified"]]),
                          ("full field", res)):
        order = subset.sort_values("Position")
        codes = [c for c in order["Abbreviation"] if c in drivers]
        ix = [drivers.index(c) for c in codes]
        pos = order.loc[order["Abbreviation"].isin(codes), "Position"].to_numpy(float)

        p = probs_all[np.ix_(ix, range(len(drivers)))][:, : len(codes)]
        p = p / p.sum(axis=1, keepdims=True)

        s = score(p, pos)
        s.update({"scored": label, "model": "published forecast", "n": len(codes)})
        rows.append(s)

        # Baseline on the grid that actually formed — the fair comparison.
        g = np.array([actual_grid[c] for c in codes], dtype=float)
        adv = grid_advantage(g)
        b = pl_probs(1.75 * (adv - adv.mean()))
        sb = score(b, pos)
        sb.update({"scored": label, "model": "baseline: actual grid", "n": len(codes)})
        rows.append(sb)

    out = pd.DataFrame(rows)[["scored", "model", "n", "rps", "ll_win", "ll_podium",
                              "ll_points", "spearman"]]
    print(out.to_string(index=False))

    # ---- headline calls ----------------------------------------------------------------
    print("\n=== headline calls ===")
    winner = res.loc[res["Position"] == 1, "Abbreviation"].iloc[0]
    fav = fc.sort_values("p_win", ascending=False).iloc[0]
    print(f"  winner: {winner}   forecast favourite: {fav.driver} at {100*fav.p_win:.1f}%"
          f"   -> {'HIT' if winner == fav.driver else 'MISS'}")
    podium = set(res.loc[res["Position"] <= 3, "Abbreviation"])
    top3 = set(fc.sort_values("p_podium", ascending=False).head(3)["driver"])
    print(f"  actual podium: {sorted(podium)}   top-3 by p_podium: {sorted(top3)}"
          f"   -> {len(podium & top3)}/3")
    dnf = res.loc[~res["classified"], "Abbreviation"].tolist()
    print(f"  retirements: {dnf}  ({100*(~res['classified']).mean():.1f}% of the field)")
    print("  the model assigned these a finishing distribution anyway — it has no "
          "reliability process")

    REPORTS.mkdir(exist_ok=True)
    out.to_csv(REPORTS / f"race_score_{args.season}_R{args.round:02d}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
