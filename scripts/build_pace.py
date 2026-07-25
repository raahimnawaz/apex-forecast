"""Fit Layer 0 pace deconvolution across a season and persist the results."""

from __future__ import annotations

import argparse

import pandas as pd

from apex.pace import fit_season, season_pace_table
from apex.paths import PROCESSED, REPORTS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    args = ap.parse_args()

    laps = pd.read_parquet(PROCESSED / f"laps_{args.season}.parquet")
    print(f"{len(laps):,} laps loaded\n")

    events = fit_season(laps, args.season)
    if not events:
        print("No events could be fitted.")
        return 1

    pace = season_pace_table(events)
    pace.to_parquet(PROCESSED / f"pace_{args.season}.parquet", index=False)

    deg_frames = []
    meta_rows = []
    for ep in events:
        d = ep.deg.copy()
        d["season"], d["round"], d["event"] = ep.season, ep.round, ep.event
        deg_frames.append(d)
        meta_rows.append({
            "season": ep.season, "round": ep.round, "event": ep.event,
            "baseline_s": round(ep.baseline_s, 3),
            "lap_trend_s_per_lap": round(ep.lap_trend_s, 4),
            "dirty_air_cost_s": round(ep.dirty_air_s, 3),
            "resid_sd_s": round(ep.resid_sd_s, 3),
            "n_laps": ep.n_laps,
            "n_drivers": ep.diagnostics["n_drivers"],
            "pseudo_r2": ep.diagnostics["pseudo_r2"],
        })

    deg = pd.concat(deg_frames, ignore_index=True)
    deg.to_parquet(PROCESSED / f"degradation_{args.season}.parquet", index=False)
    meta = pd.DataFrame(meta_rows)
    meta.to_csv(REPORTS / f"pace_fit_{args.season}.csv", index=False)

    print("=== per-event fit diagnostics ===")
    print(meta.to_string(index=False))

    print("\n=== season-average corrected pace (s vs field mean, negative = faster) ===")
    agg = (pace.groupby(["Driver", "Team"])
                .agg(pace_s=("pace_s", "mean"), sd=("pace_s", "std"),
                     races=("round", "nunique"), laps=("n_laps", "sum"))
                .sort_values("pace_s").round(3).reset_index())
    print(agg.to_string(index=False))

    print("\n=== degradation by compound (s/lap) ===")
    print(deg.groupby("compound")
             .agg(deg=("deg_s_per_lap", "mean"), sd=("deg_s_per_lap", "std"), n=("n_laps", "sum"))
             .round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
