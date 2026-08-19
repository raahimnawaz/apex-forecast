#!/bin/bash
#
# Weekly local refresh: ingest the race that just ran, refit, publish, push.
#
# This runs on this machine rather than in GitHub Actions because
# livetiming.formula1.com answers 403 to datacenter IPs — see the header of
# .github/workflows/update.yml for the measurement. A residential connection is the
# only place the ingest can happen at all.
#
# Pushing web/data is what updates the live dashboard: pages.yml triggers on web/**.
#
# Scheduled by ops/com.raahimnawaz.apex-refresh.plist. Logs to reports/refresh.log,
# which is gitignored (reports/*.log) and is the first place to look if a round is
# missing from the calibration sample.

set -uo pipefail

REPO="$HOME/apex-forecast"
PY="$REPO/.venv/bin/python"
LOG="$REPO/reports/refresh.log"

cd "$REPO" || { echo "no repo at $REPO"; exit 1; }
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

echo
echo "================ $(date '+%Y-%m-%d %H:%M:%S %Z') ================"

# Never refresh on top of a dirty tree — a half-finished edit would be committed
# alongside the model output and attributed to the refresh.
if [ -n "$(git status --porcelain -- src scripts ops Makefile pyproject.toml)" ]; then
  echo "ABORT: uncommitted source changes; refusing to auto-commit on top of them"
  git status --short -- src scripts ops Makefile pyproject.toml
  exit 1
fi

git pull --ff-only origin main || { echo "ABORT: pull is not a fast-forward"; exit 1; }

# The canary distinguishes three outcomes; only one of them is a defect here.
make spike
code=$?
case "$code" in
  0) echo "spike: feed is readable" ;;
  2) echo "SKIP: live timing refused this machine — nothing fetched, exiting quietly"
     exit 0 ;;
  *) echo "FAIL: spike exit $code — the feed served data that will not parse. Look at it."
     exit "$code" ;;
esac

make all || { echo "FAIL: make all"; exit 1; }

# Score the round that just ran against the forecast published before it. score_race.py
# refuses when no prediction log exists for that round, which is normal early in a
# season and must not fail the refresh.
rnd=$("$PY" -c "
import fastf1, pandas as pd
fastf1.Cache.enable_cache('data/cache')
s = fastf1.get_event_schedule(2026, include_testing=False)
done = s[s['EventDate'] < pd.Timestamp.now().normalize()]
print(int(done.iloc[-1]['RoundNumber']) if len(done) else '')
" 2>/dev/null)
if [ -n "$rnd" ]; then
  echo "scoring R$rnd (a refusal here is expected when no prediction log exists)"
  "$PY" scripts/score_race.py --round "$rnd" || echo "  not scored"
fi

make test lint || { echo "FAIL: tests or lint — not publishing"; exit 1; }

# data/ is gitignored and reproducible; only the published payloads, the write-once
# prediction log and the reports are worth committing.
git add web/data reports

# Every build restamps generated_utc even when the output is otherwise identical, and
# news items whose feed omits a pubDate are stamped with now() on every fetch (see the
# news-dating limitation in docs/HANDOFF.md), so a plain "did anything change" test is
# always true. Left alone that produces a no-op commit every week, a pointless dashboard
# redeploy, and a message claiming a refresh that did not happen.
#
# Real news arrives as new titles, links and summaries, so ignoring the two timestamp
# fields still detects it; what it stops detecting is a clock ticking.
substantive=$(git diff --staged -U0 \
  | grep -E '^[+-]' | grep -vE '^(\+\+\+|---)' \
  | grep -v '"generated_utc"' | grep -vc '"published"')

if git diff --staged --quiet; then
  echo "nothing changed"
elif [ "$substantive" -eq 0 ]; then
  echo "only timestamps changed — nothing to publish"
  git restore --staged --worktree web/data reports
else
  # Every fit is seeded, so an unchanged season refits byte-identically and only the
  # headlines move. Say which of the two actually happened rather than claiming a model
  # refresh on a week where nothing raced.
  if git diff --staged --name-only | grep -qv '^web/data/news.json$'; then
    msg="Refresh the model and forecast the next round"
  else
    msg="Refresh headlines"
  fi
  git commit -m "$msg" || exit 1
  git push origin main || { echo "FAIL: push"; exit 1; }
  echo "pushed $(git rev-parse --short HEAD) — dashboard redeploys via pages.yml"
fi

"$PY" scripts/status.py --season 2026
echo "done $(date '+%H:%M:%S')"
