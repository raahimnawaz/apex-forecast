"""Score a published forecast against the race that actually happened.

This is the only test that cannot be gamed: the forecast was written and committed before
the race existed. Nothing here refits anything — it reads what was published and compares
it to the result.

That claim depends entirely on reading the right file. `strength_{season}.json` always
holds the *next* race and is overwritten on every build, so scoring a past round against it
silently grades a different forecast and prints confident numbers for it. The forecast is
therefore read from `web/data/predictions/{season}_R{round}.json`, the write-once
prediction log, and this script refuses to score at all rather than fall back to a payload
that belongs to another round.

Scored two ways on purpose:
  classified — against finishers only, which is the pace model's own claim
  full       — against the whole field including retirements, which is how a reader
               would actually use the numbers

The gap between those two is what the reliability layer has to earn. Before that layer
existed the full-field score was strictly the worse of the two; it is now the forecast's
to win or lose on its own terms.
"""

from __future__ import annotations

import argparse
import json

import fastf1
import numpy as np
import pandas as pd

from apex.paths import CACHE, REPORTS, WEB_DATA
from apex.scoring import score
from apex.strength import grid_advantage

fastf1.Cache.enable_cache(str(CACHE))


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

    logged = WEB_DATA / "predictions" / f"{args.season}_R{args.round:02d}.json"
    if logged.exists():
        published = json.loads(logged.read_text())
        src = f"prediction log {logged.name}"
    else:
        # Falling back to the live payload used to be the default, and it is a trap:
        # strength_{season}.json always holds the *next* race, so after the following
        # round is built this silently grades the wrong forecast and prints confident
        # numbers for it. Refuse instead of guessing.
        live = json.loads((WEB_DATA / f"strength_{args.season}.json").read_text())
        live_round = (live.get("diagnostics") or {}).get("forecast_round")
        if live_round != args.round:
            raise SystemExit(
                f"no prediction log for {args.season} R{args.round}, and the live payload "
                f"holds round {live_round}. Scoring it would grade the wrong forecast. "
                f"Expected {logged}."
            )
        published = live
        src = f"live payload (round {live_round})"

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
          f"{published['generated_utc']} ===")
    print(f"    source: {src}\n")

    # ---- the grid the model assumed vs the grid that actually formed -------------------
    # A forecast published before qualifying has no grid at all, which is the normal state
    # for a next-race forecast rather than an error. Entries without one are skipped: there
    # is no assumed position to compare against, and coercing None to an int here used to
    # abort the whole scoring run over a section that is only commentary.
    q = fc.set_index("driver")["grid"].to_dict()
    actual_grid = res.set_index("Abbreviation")["GridPosition"].to_dict()
    known = {d: v for d, v in q.items() if v is not None and np.isfinite(float(v))}
    if not known:
        print("the published forecast carried no grid — it was made before qualifying\n")
    moved = [(d, int(known[d]), int(actual_grid[d]))
             for d in known
             if d in actual_grid and int(known[d]) != int(actual_grid[d])]
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

        s = score(p, pos, digits=4)
        s.update({"scored": label, "model": "published forecast", "n": len(codes)})
        rows.append(s)

        # Baseline on the grid that actually formed — the fair comparison.
        g = np.array([actual_grid[c] for c in codes], dtype=float)
        adv = grid_advantage(g)
        b = pl_probs(1.75 * (adv - adv.mean()))
        sb = score(b, pos, digits=4)
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
    # The reliability layer publishes a per-car finishing probability, so the retirements
    # are a scoreable call rather than something the forecast was silent about. If the
    # cars that stopped were not rated riskier than the field, the layer added nothing here.
    if "p_finish" in fc.columns and dnf:
        risk = (1.0 - fc.set_index("driver")["p_finish"]).dropna()
        hit = risk.reindex(dnf).dropna()
        if not hit.empty:
            print(f"  forecast retirement risk: {100*hit.mean():.1f}% for the cars that "
                  f"stopped vs {100*risk.mean():.1f}% across the field")

    REPORTS.mkdir(exist_ok=True)
    out.to_csv(REPORTS / f"race_score_{args.season}_R{args.round:02d}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
