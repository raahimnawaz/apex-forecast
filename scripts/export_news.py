"""Refresh the news payload.

Deliberately separate from export_web.py: headlines change hourly, the models do not.
This runs in a couple of seconds and needs no fitted posterior, so the news section can
be refreshed on its own schedule without touching anything the models produced.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from apex.news import FEEDS, fetch_all
from apex.paths import WEB_DATA


def main() -> int:
    print(f"fetching {len(FEEDS)} feeds")
    items = fetch_all()
    if not items:
        print("no items fetched — leaving the existing payload alone")
        return 1

    tagged = sum(1 for i in items if i["teams"])
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": [s for s, _ in FEEDS],
        "items": items,
    }
    out = WEB_DATA / "news.json"
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    print(f"  {len(items)} headlines, {tagged} tagged to a team, "
          f"newest {items[0]['published']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
