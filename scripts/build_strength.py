"""Fit Layer 1 and write the strength posterior + next-race forecast."""

from __future__ import annotations

import argparse
import glob
import pickle

import numpy as np
import pandas as pd

from apex import weather as wx
from apex.championship import FAN_QUANTILES, project_championship
from apex.history import fetch_season_results
from apex.paths import GRIDS, PROCESSED, RAW, REPORTS
from apex.reliability import fit_reliability, simulate_race
from apex.strength import build, fit, predict_order

# Chosen by the walk-forward bake-off, not by preference: attrition beat the forward
# and contaminated likelihoods and the grid baseline on every metric.
LIKELIHOOD = "attrition"


def load_2026_results() -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW / "results_2026_R*_R.parquet")))
    if not files:
        raise SystemExit("no 2026 race results — run scripts/ingest.py first")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_2026_sprints() -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW / "results_2026_R*_S.parquet")))
    return (pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            if files else pd.DataFrame())


def remaining_rounds(season: int, from_round: int) -> list[tuple[int, bool]]:
    """[(round, is_sprint), ...] for every round still to run, in order.

    The sprint flag comes from the schedule's EventFormat rather than from a hardcoded
    list, because it decides how many points are still on the table: 2026 runs six sprint
    weekends and two of them are still to come, worth 16 points on top of the 300.
    """
    import fastf1

    from apex.paths import CACHE
    fastf1.Cache.enable_cache(str(CACHE))
    sched = fastf1.get_event_schedule(season, include_testing=False)
    rows = sched[sched["RoundNumber"] >= from_round].sort_values("RoundNumber")
    return [(int(r["RoundNumber"]),
             "sprint" in str(r["EventFormat"]).lower()) for _, r in rows.iterrows()]


def season_points(r26: pd.DataFrame, sprints: pd.DataFrame) -> tuple[dict, dict]:
    """Driver and constructor points so far, sprints included.

    Kept separate rather than summed one from the other: 24 drivers have held 22 seats
    this season, so a driver's points follow the driver while the constructor points they
    earned stay with the team that earned them.
    """
    frames = [r26[["Abbreviation", "TeamName", "Points"]]]
    if not sprints.empty:
        frames.append(sprints[["Abbreviation", "TeamName", "Points"]])
    allp = pd.concat(frames, ignore_index=True)
    allp["Points"] = pd.to_numeric(allp["Points"], errors="coerce").fillna(0.0)
    return (allp.groupby("Abbreviation")["Points"].sum().to_dict(),
            allp.groupby("TeamName")["Points"].sum().to_dict())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    args = ap.parse_args()

    r25 = fetch_season_results(2025)
    r26 = load_2026_results()
    sprints = load_2026_sprints()
    d = build(r25, r26, sprints_2026=sprints)

    n_sprint = int(d.races["event"].str.contains(r"\(sprint\)").sum())
    print(f"races: {len(d.races)}  ({(d.era == 0).sum()} in 2025, "
          f"{(d.era == 1).sum()} in 2026, of which {n_sprint} sprints)")
    print(f"circuits: {d.n_circuits}")
    print(f"drivers: {len(d.drivers)}   constructors: {len(d.constructors)}")
    shared = set(r25["code"]) & set(r26["Abbreviation"])
    print(f"drivers present in both seasons: {len(shared)}  -> these tie the eras together")
    print()

    # Attrition (reverse Plackett-Luce) is the production likelihood because it won the
    # walk-forward bake-off on every metric — see reports/calibration_summary.csv. It is
    # also the right choice for the driver-versus-car split specifically: the forward
    # model let three incident races drag Mercedes' car term down, which is the exact
    # pathology Graves et al. (2003) designed the reverse model to avoid.
    mcmc = fit(d, warmup=args.warmup, samples=args.samples, chains=args.chains,
               likelihood=LIKELIHOOD)
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

    # --- Tier 1: reliability, so the forecast covers the whole field -------------------
    laps_path = PROCESSED / "laps_2026.parquet"
    laps = pd.read_parquet(laps_path) if laps_path.exists() else None
    rel = fit_reliability(r25, r26, laps_2026=laps)
    print(f"\n=== retirement risk (hierarchical, pooled toward the grid mean "
          f"{100 * rel.grid_mean:.1f}%) ===")
    print(rel.by_entry.head(8).round(3).to_string(index=False))
    print(f"\nsafety car in {100 * rel.sc_rate:.0f}% of races, "
          f"VSC in {100 * rel.vsc_rate:.0f}% (only the safety car is simulated)")

    # --- next-race forecast ------------------------------------------------------------
    next_round = int(r26["round"].max()) + 1
    quali = RAW / f"results_2026_R{next_round:02d}_Q.parquet"
    race_res = RAW / f"results_2026_R{next_round:02d}_R.parquet"
    entries, grid, grid_source = None, None, "none"

    # The actual starting grid is not the qualifying classification. At round 11 six
    # drivers started somewhere other than where they qualified — Hamilton P2 to P5,
    # Antonelli P4 to P7 — and forecasting off the classification quietly got those
    # wrong. Prefer the real grid whenever it exists.
    override = GRIDS / f"2026_R{next_round:02d}.csv"

    if race_res.exists():
        rr = pd.read_parquet(race_res)
        rr = rr[pd.to_numeric(rr["GridPosition"], errors="coerce").notna()]
        entries = [(r.Abbreviation, r.TeamName) for r in rr.itertuples()]
        grid = pd.to_numeric(rr["GridPosition"]).astype(float).tolist()
        grid_source = f"R{next_round} actual starting grid"
    elif override.exists() and quali.exists():
        # Penalties are published on Saturday evening but appear in no machine-readable
        # feed this project can reach: FastF1 leaves GridPosition empty on a qualifying
        # session and only fills it in on the race session, which does not exist until the
        # race has run. So the corrected grid is typed in by hand and read here.
        q = pd.read_parquet(quali).sort_values("Position")
        ov = pd.read_csv(override)
        missing = set(q["Abbreviation"]) - set(ov["driver"])
        if missing:
            raise SystemExit(f"{override.name} is missing {sorted(missing)} — a partial "
                             f"grid would silently mix penalised and unpenalised positions")
        pos = ov.set_index("driver")["grid"].astype(float)
        q = q.assign(_g=q["Abbreviation"].map(pos)).sort_values("_g")
        entries = [(r.Abbreviation, r.TeamName) for r in q.itertuples()]
        grid = q["_g"].astype(float).tolist()
        moved = int((q["_g"].to_numpy() != q["Position"].to_numpy()).sum())
        grid_source = (f"R{next_round} starting grid from {override.name} "
                       f"({moved} of {len(q)} moved by penalty)")
    elif quali.exists():
        q = pd.read_parquet(quali).sort_values("Position")
        entries = [(r.Abbreviation, r.TeamName) for r in q.itertuples()]
        grid = q["Position"].astype(float).tolist()
        grid_source = (f"R{next_round} qualifying classification "
                       f"(grid penalties NOT applied)")
        # This is the error that cost the round 11 forecast: six drivers started somewhere
        # other than where they qualified, Hamilton P2 -> P5 and Antonelli P4 -> P7. Across
        # 2026 it is 16% of entries in 5 of 11 races, and once 20 of 22 cars. Loud, because
        # the forecast is still perfectly usable and the fix takes two minutes.
        print("\n  !! WARNING: forecasting off the qualifying classification.")
        print("     Grid penalties are NOT applied. Historically this moves ~16% of the")
        print(f"     field. Write {override} with columns driver,grid to correct it.")

    circuit = None
    try:
        import fastf1

        from apex.paths import CACHE
        fastf1.Cache.enable_cache(str(CACHE))
        circuit = str(fastf1.get_event(2026, next_round)["EventName"])
    except Exception as exc:  # noqa: BLE001 - a missing schedule must not stop the build
        print(f"  circuit lookup failed ({type(exc).__name__}); "
              "forecasting without a circuit-specific grid effect")

    beta = beta_lo = beta_hi = circ_mult = None
    if grid is not None:
        print(f"\ngrid known ({grid_source}); fitting grid-conditional model")
        mcmc_g = fit(d, warmup=args.warmup, samples=args.samples, chains=args.chains,
                     use_grid=True, likelihood=LIKELIHOOD)
        post_g = mcmc_g.get_samples()
        beta = float(np.mean(post_g["beta_grid"]))
        beta_lo, beta_hi = np.percentile(post_g["beta_grid"], [5.5, 94.5])
        print(f"  grid advantage beta = {beta:.3f}  (89% CI {beta_lo:.3f} … {beta_hi:.3f})")
        if circuit in d.circuits and "circuit_mult" in post_g:
            circ_mult = float(np.mean(np.asarray(post_g["circuit_mult"])
                                      [:, d.circuits.index(circuit)]))
            print(f"  circuit multiplier for {circuit}: {circ_mult:.3f} "
                  f"-> effective beta {beta * circ_mult:.3f}")
        use_post, use_grid_arg = post_g, grid
    else:
        latest_race = r26[r26["round"] == r26["round"].max()]
        entries = sorted({(r.Abbreviation, r.TeamName) for r in latest_race.itertuples()})
        print("\nno grid yet for the next round; forecasting without a grid term")
        use_post, use_grid_arg = post, None

    _, theta = predict_order(use_post, d, entries, grid=use_grid_arg,
                             likelihood=LIKELIHOOD, circuit=circuit)

    # --- Tier 2: weather ----------------------------------------------------------------
    ev_date = ""
    try:
        ev_date = pd.Timestamp(fastf1.get_event(2026, next_round)["EventDate"]).strftime("%Y-%m-%d")
    except Exception as exc:  # noqa: BLE001
        print(f"  event date lookup failed ({type(exc).__name__}); skipping weather")
    weather = wx.fetch(circuit or "", ev_date) if ev_date else wx.RaceWeather("", "", False)
    wmult = wx.uncertainty_multiplier(weather)
    if weather.available:
        print(f"\nweather: rain probability {100 * (weather.rain_probability or 0):.0f}%, "
              f"{weather.temperature_c} C -> uncertainty x{wmult}")

    # --- full-field simulation: retirements + safety cars + weather --------------------
    p_dnf = np.array([
        float(rel.by_entry.loc[(rel.by_entry.driver == c) & (rel.by_entry.team == t),
                               "p_dnf"].iloc[0])
        if ((rel.by_entry.driver == c) & (rel.by_entry.team == t)).any() else rel.grid_mean
        for c, t in entries])

    sc = rel.sc_rate
    if circuit and not rel.fcy_by_circuit.empty:
        row = rel.fcy_by_circuit[rel.fcy_by_circuit["event"] == circuit]
        if not row.empty:
            sc = float(row["sc_rate"].iloc[0])

    probs = simulate_race(theta / wmult, p_dnf, sc, likelihood=LIKELIHOOD)
    print(f"\nsimulated: per-car retirement risk + safety car at {100 * sc:.0f}%")

    # --- championship projection (PLAN.md section 3, view 5) --------------------------
    # Deliberately off the *grid-free* posterior and without the circuit or weather terms:
    # none of qualifying, circuit or weather is known for December, and a projection that
    # borrowed this weekend's values for all twelve rounds would be a different claim than
    # the one it appears to make. See src/apex/championship.py.
    champ = None
    try:
        remaining = remaining_rounds(2026, next_round)
        drv_pts, team_pts = season_points(r26, sprints)
        _, theta_free = predict_order(post, d, entries, grid=None,
                                      likelihood=LIKELIHOOD, circuit=None)
        champ = project_championship(
            theta_free, p_dnf, rel.sc_rate, entries,
            np.array([float(drv_pts.get(c, 0.0)) for c, _ in entries]),
            remaining, team_points_now=team_pts, likelihood=LIKELIHOOD)
        n_sp = sum(1 for _, sp in remaining if sp)
        print(f"\n=== championship: {len(remaining)} rounds left "
              f"({n_sp} sprints), {champ.n_sim} seasons simulated ===")
        lead = np.argsort(-champ.title_prob)[:5]
        print(pd.DataFrame({
            "driver": [champ.drivers[i] for i in lead],
            "points": [champ.points_now[i] for i in lead],
            "title%": [round(100 * champ.title_prob[i], 1) for i in lead],
            "exp_final": [round(champ.exp_points[i]) for i in lead],
        }).to_string(index=False))
        print(f"  still mathematically possible: {int(champ.still_possible.sum())} "
              f"of {len(entries)}")

        import json as _cj
        order = np.argsort(-champ.title_prob)
        (PROCESSED / "championship.json").write_text(_cj.dumps({
            "rounds": champ.rounds,
            "n_sim": champ.n_sim,
            "quantiles": list(FAN_QUANTILES),
            "sprints_remaining": n_sp,
            "drivers": [{
                "driver": champ.drivers[i], "team": champ.teams[i],
                "points_now": round(float(champ.points_now[i]), 1),
                "title_prob": round(float(champ.title_prob[i]), 5),
                "exp_points": round(float(champ.exp_points[i]), 1),
                "still_possible": bool(champ.still_possible[i]),
                # The fan is only carried for drivers who can still win it; for anyone
                # else it is a band around a number the page does not plot.
                "fan": ([[round(float(v), 1) for v in champ.fan[q, :, i]]
                         for q in range(len(FAN_QUANTILES))]
                        if champ.still_possible[i] else None),
            } for i in order],
            "constructors": [{
                "team": champ.team_names[j],
                "points_now": round(float(champ.team_points_now[j]), 1),
                "title_prob": round(float(champ.team_title_prob[j]), 5),
                "exp_points": round(float(champ.team_exp_points[j]), 1),
            } for j in np.argsort(-champ.team_title_prob)],
        }))
    except Exception as exc:  # noqa: BLE001 - a schedule lookup must not stop the build
        print(f"\nchampionship projection skipped ({type(exc).__name__}: {exc})")

    fc = pd.DataFrame({
        "driver": [e[0] for e in entries],
        "team": [e[1] for e in entries],
        "grid": grid if grid is not None else [None] * len(entries),
        "p_finish": 1.0 - p_dnf,
        "p_win": probs[:, 0],
        "p_podium": probs[:, :3].sum(1),
        "p_points": probs[:, :10].sum(1),
        "exp_pos": (probs * np.arange(1, len(entries) + 1)).sum(1),
    }).sort_values("p_win", ascending=False).reset_index(drop=True)

    print(f"\n=== round {next_round} forecast (full field, retirements included; grid: {grid_source}) ===")
    print(fc.round(4).to_string(index=False))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    sk.to_parquet(PROCESSED / "skill_2026.parquet", index=False)
    cs.to_parquet(PROCESSED / "constructor_2026.parquet", index=False)
    fc.to_parquet(PROCESSED / "forecast_next.parquet", index=False)
    np.save(PROCESSED / "position_probs.npy", probs)
    pd.DataFrame({"driver": [e[0] for e in entries], "team": [e[1] for e in entries],
                  "grid": grid if grid is not None else [None] * len(entries)}) \
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
        "likelihood": LIKELIHOOD,
        "forecast_round": next_round,
        "grid_source": grid_source,
        "beta_grid": None if beta is None else round(beta, 4),
        "circuit": circuit,
        "circuit_mult": None if circ_mult is None else round(circ_mult, 4),
        "sc_rate": round(sc, 4),
        "vsc_rate": round(rel.vsc_rate, 4),
        "grid_mean_dnf": round(rel.grid_mean, 4),
        "weather_multiplier": wmult,
        "n_sprints": n_sprint,
        "beta_grid_lo": None if beta is None else round(float(beta_lo), 4),
        "beta_grid_hi": None if beta is None else round(float(beta_hi), 4),
    }
    rel.by_entry.to_parquet(PROCESSED / "reliability.parquet", index=False)
    import json as _json
    (PROCESSED / "weather_next.json").write_text(_json.dumps(wx.to_dict(weather)))
    pd.DataFrame([diag]).to_csv(REPORTS / "strength_fit.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
