"""Derive the hero artwork from the fastest qualifying lap of a round.

Separate from export_web.py because it needs telemetry, which is a much heavier FastF1
load than the timing data everything else uses, and it only changes once per weekend.
"""

from __future__ import annotations

import argparse
import json

import fastf1

from apex.paths import CACHE, WEB_DATA
from apex.trackart import build, to_dict

fastf1.Cache.enable_cache(str(CACHE))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--session", default="Q")
    args = ap.parse_args()

    ses = fastf1.get_session(args.season, args.round, args.session)
    ses.load(telemetry=True, weather=False, messages=False)

    art = to_dict(build(ses))
    out = WEB_DATA / f"trackart_{args.season}_R{args.round:02d}.json"
    out.write_text(json.dumps(art))

    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    print(f"  {art['circuit']} · {art['driver']} {art['lap_time']} · "
          f"{len(art['points'])} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
