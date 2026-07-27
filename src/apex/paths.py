"""Canonical project paths. Everything else imports from here."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = DATA / "cache"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
WEB = ROOT / "web"
WEB_DATA = WEB / "data"

# Hand-entered starting grids, and the one data directory that is *not* under `data/`.
# Everything in `data/` is gitignored because it can be rebuilt from FastF1 and Jolpica;
# a grid corrected for penalties cannot be, because no upstream source publishes it before
# the race. It is typed in from the stewards' decisions, so it is source, not cache.
GRIDS = ROOT / "grids"

for _p in (CACHE, RAW, PROCESSED, REPORTS, WEB_DATA, GRIDS):
    _p.mkdir(parents=True, exist_ok=True)
