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

for _p in (CACHE, RAW, PROCESSED, REPORTS, WEB_DATA):
    _p.mkdir(parents=True, exist_ok=True)
