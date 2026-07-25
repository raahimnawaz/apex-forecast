"""Ingest a season into the raw parquet store and write a data-quality report."""

from __future__ import annotations

import argparse

import pandas as pd

from apex.ingest import ingest_season
from apex.paths import PROCESSED, REPORTS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--through-round", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    print(f"Ingesting {args.season}\n")
    laps, qual = ingest_season(args.season, through_round=args.through_round, force=args.force)

    if laps.empty:
        print("\nNo laps ingested.")
        return 1

    out = PROCESSED / f"laps_{args.season}.parquet"
    laps.to_parquet(out, index=False)

    print(f"\n{len(laps):,} laps -> {out}")
    print(f"rounds: {sorted(laps['round'].unique())}")
    print(f"sessions: {sorted(laps['session'].unique())}")
    print(f"drivers: {laps['Driver'].nunique()}  teams: {laps['Team'].nunique()}")

    if not qual.empty:
        qpath = REPORTS / f"data_quality_{args.season}.csv"
        qual.to_csv(qpath, index=False)
        bad = qual[~qual["ok"]]
        print(f"\nquality report -> {qpath}")
        print(f"sessions ingested: {len(qual)}   flagged: {len(bad)}")
        if not bad.empty:
            with pd.option_context("display.width", 200, "display.max_colwidth", 60):
                print(bad[["round", "event", "session", "n_laps", "n_accurate",
                           "median_lap_s", "ghost_drivers", "note"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
