"""Fit Layer 1 and write the strength posterior + next-race forecast."""

from __future__ import annotations

import argparse
import glob
import pickle

import numpy as np
import pandas as pd

from apex.history import fetch_season_results
from apex.paths import PROCESSED, RAW, REPORTS
from apex.strength import build, fit, predict_order


def load_2026_results() -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW / "results_2026_R*_R.parquet")))
    if not files:
        raise SystemExit("no 2026 race results — run scripts/ingest.py first")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    args = ap.parse_args()

    r25 = fetch_season_results(2025)
    r26 = load_2026_results()
    d = build(r25, r26)

    print(f"races: {len(d.races)}  ({(d.era == 0).sum()} in 2025, {(d.era == 1).sum()} in 2026)")
    print(f"drivers: {len(d.drivers)}   constructors: {len(d.constructors)}")
    shared = set(r25["code"]) & set(r26["Abbreviation"])
    print(f"drivers present in both seasons: {len(shared)}  -> these tie the eras together")
    print()

    mcmc = fit(d, warmup=args.warmup, samples=args.samples, chains=args.chains)
    mcmc.print_summary(exclude_deterministic=True)
    post = mcmc.get_samples()

    # --- convergence: a posterior nobody checked is a number nobody should trust -------
    import numpyro.diagnostics as diag_mod
    grouped = mcmc.get_samples(group_by_chain=True)
    worst_rhat, worst_site = 0.0, ""
    for site, arr in grouped.items():
        if site.endswith("_raw") or arr.ndim < 2:
            continue
        rh = np.asarray(diag_mod.gelman_rubin(np.asarray(arr))) if arr.shape[0] > 1 else np.array([1.0])
        if np.nanmax(rh) > worst_rhat:
            worst_rhat, worst_site = float(np.nanmax(rh)), site
    n_div = int(mcmc.get_extra_fields().get("diverging", np.array([])).sum()) \
        if "diverging" in mcmc.get_extra_fields() else -1
    print(f"\nworst R-hat: {worst_rhat:.4f} ({worst_site})   divergences: "
          f"{'n/a' if n_div < 0 else n_div}")
    if worst_rhat > 1.05:
        print("  WARNING: R-hat above 1.05 — treat these posteriors as unconverged")

    # --- driver skill ------------------------------------------------------------------
    skill = np.asarray(post["skill"])
    sk = pd.DataFrame({
        "driver": d.drivers,
        "skill": skill.mean(0),
        "lo": np.percentile(skill, 5.5, axis=0),
        "hi": np.percentile(skill, 94.5, axis=0),
    }).sort_values("skill", ascending=False).reset_index(drop=True)

    # --- constructor strength ----------------------------------------------------------
    car26 = np.asarray(post["car26"])            # (S, n_c, n_t)
    car25 = np.asarray(post["car25"])
    latest = car26[:, :, -1]
    cs = pd.DataFrame({
        "constructor": d.constructors,
        "car_2026_latest": latest.mean(0),
        "lo": np.percentile(latest, 5.5, axis=0),
        "hi": np.percentile(latest, 94.5, axis=0),
        "car_2026_r1": car26[:, :, 0].mean(0),
        "car_2025": car25.mean(0),
    })
    cs["development"] = cs["car_2026_latest"] - cs["car_2026_r1"]
    cs = cs.sort_values("car_2026_latest", ascending=False).reset_index(drop=True)

    print("\n=== driver skill (log-odds, higher = better; 89% CI) ===")
    print(sk.round(3).to_string(index=False))
    print("\n=== constructor strength, 2026 (log-odds; development = R10 minus R1) ===")
    print(cs.round(3).to_string(index=False))

    # --- how much of the spread is the car? --------------------------------------------
    var_skill = float(np.var(skill.mean(0)))
    var_car = float(np.var(latest.mean(0)))
    share = var_car / (var_car + var_skill)
    print(f"\nconstructor share of explained spread: {100 * share:.1f}%")

    # --- independent cross-check against Layer 0 ---------------------------------------
    # Layer 0 measures pace from lap times; Layer 1 infers strength from finishing order.
    # They share no likelihood, no data representation and no fitting method, so their
    # agreement is genuine evidence rather than a consistency check of one model.
    rho = None
    pace_path = PROCESSED / "pace_2026.parquet"
    if pace_path.exists():
        from scipy.stats import spearmanr
        team_pace = pd.read_parquet(pace_path).groupby("Team")["pace_s"].mean()
        m = cs.set_index("constructor").join(team_pace).dropna(subset=["pace_s"])
        if len(m) > 3:
            rho = float(spearmanr(m["car_2026_latest"], -m["pace_s"]).statistic)
            print(f"cross-check vs Layer 0 corrected pace: Spearman rho = {rho:.3f} "
                  f"over {len(m)} constructors")

    # --- next-race forecast ------------------------------------------------------------
    latest_race = r26[r26["round"] == r26["round"].max()]
    entries = sorted({(r.Abbreviation, r.TeamName) for r in latest_race.itertuples()})
    probs, _theta = predict_order(post, d, entries)

    fc = pd.DataFrame({
        "driver": [e[0] for e in entries],
        "team": [e[1] for e in entries],
        "p_win": probs[:, 0],
        "p_podium": probs[:, :3].sum(1),
        "p_points": probs[:, :10].sum(1),
        "exp_pos": (probs * np.arange(1, len(entries) + 1)).sum(1),
    }).sort_values("p_win", ascending=False).reset_index(drop=True)

    print("\n=== next-race forecast (conditional on finishing) ===")
    print(fc.round(4).to_string(index=False))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    sk.to_parquet(PROCESSED / "skill_2026.parquet", index=False)
    cs.to_parquet(PROCESSED / "constructor_2026.parquet", index=False)
    fc.to_parquet(PROCESSED / "forecast_next.parquet", index=False)
    np.save(PROCESSED / "position_probs.npy", probs)
    pd.DataFrame({"driver": [e[0] for e in entries], "team": [e[1] for e in entries]}) \
        .to_parquet(PROCESSED / "forecast_entries.parquet", index=False)

    car_hist = []
    for i, c in enumerate(d.constructors):
        for t in range(car26.shape[2]):
            car_hist.append({"constructor": c, "round": t + 1,
                             "strength": float(car26[:, i, t].mean()),
                             "lo": float(np.percentile(car26[:, i, t], 5.5)),
                             "hi": float(np.percentile(car26[:, i, t], 94.5))})
    pd.DataFrame(car_hist).to_parquet(PROCESSED / "constructor_by_round.parquet", index=False)

    with open(PROCESSED / "strength_meta.pkl", "wb") as f:
        pickle.dump({"drivers": d.drivers, "constructors": d.constructors,
                     "n_races": len(d.races), "constructor_share": share}, f)

    diag = {
        "n_races": len(d.races), "n_2025": int((d.era == 0).sum()),
        "n_2026": int((d.era == 1).sum()), "n_drivers": len(d.drivers),
        "n_constructors": len(d.constructors), "shared_drivers": len(shared),
        "constructor_share": round(share, 4),
        "sigma_skill": float(np.mean(post["sigma_skill"])),
        "sigma_delta": float(np.mean(post["sigma_delta"])),
        "sigma_walk": float(np.mean(post["sigma_walk"])),
        "worst_rhat": round(worst_rhat, 4),
        "layer0_spearman": None if rho is None else round(rho, 4),
        "top_m": 10,
    }
    pd.DataFrame([diag]).to_csv(REPORTS / "strength_fit.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
