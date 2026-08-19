"""Fetch F1 news headlines from public RSS feeds at build time.

Two rules govern everything in this module.

**Nothing here is generated.** Headlines, links and timestamps come from the source
feed verbatim; the module never writes a summary of its own. If a feed is down, its
items are simply absent — an empty news section is correct, an invented one is not.

**Feed content is untrusted input, not instruction.** It is third-party text that
arrives over the network, so it is stripped of markup here and rendered with
`textContent` on the client. It is never evaluated, never interpolated into HTML, and
nothing in it is treated as a directive. Only the headline, source, timestamp and link
are kept; article bodies are not reproduced.
"""

from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime

import requests

from apex.paths import STATE

FEEDS = [
    ("Autosport", "https://www.autosport.com/rss/f1/news/"),
    ("Motorsport.com", "https://www.motorsport.com/rss/f1/news/"),
    ("BBC Sport", "https://feeds.bbci.co.uk/sport/formula1/rss.xml"),
    ("Formula1.com", "https://www.formula1.com/en/latest/all.xml"),
]

UA = {"User-Agent": "Mozilla/5.0 (compatible; apex-forecast/0.1)"}

# Links whose feed gave no usable pubDate, and the first fetch that saw each one. Only
# undated links are recorded, so the file stays small enough to commit every week.
SEEN_PATH = STATE / "news_seen.json"

# How long an undated link is remembered after it drops out of every feed. Long enough
# that a feed outage cannot make an item look new again, short enough to bound the file.
SEEN_RETENTION = timedelta(days=30)

# Team -> the strings that identify it in a headline. Driver surnames are included
# because most F1 headlines name the driver, not the constructor.
TEAM_TERMS = {
    "Mercedes": ["mercedes", "antonelli", "russell"],
    "Ferrari": ["ferrari", "hamilton", "leclerc"],
    "Red Bull Racing": ["red bull", "verstappen", "hadjar"],
    "McLaren": ["mclaren", "norris", "piastri"],
    "Aston Martin": ["aston martin", "alonso", "stroll"],
    "Alpine": ["alpine", "gasly", "colapinto"],
    "Williams": ["williams", "albon", "sainz"],
    "Racing Bulls": ["racing bulls", "lawson", "lindblad"],
    "Audi": ["audi", "sauber", "hulkenberg", "hülkenberg", "bortoleto"],
    "Haas F1 Team": ["haas", "ocon", "bearman"],
    "Cadillac": ["cadillac", "perez", "pérez", "bottas"],
}

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


@dataclass
class Item:
    title: str
    link: str
    source: str
    published: str | None   # ISO-8601 UTC; None until fetch_all resolves an undated item
    published_ts: float | None
    summary: str
    teams: list[str]
    date_estimated: bool = False   # True when the date is first-seen, not published


def _clean(text: str | None, limit: int = 260) -> str:
    """Strip markup and entities from feed text. Never rendered as HTML downstream."""
    if not text:
        return ""
    t = html.unescape(TAG_RE.sub(" ", text))
    t = WS_RE.sub(" ", t).strip()
    return t[:limit].rstrip() + ("…" if len(t) > limit else "")


def _parse_date(raw: str | None) -> datetime | None:
    """The feed's publication time, or None when it did not give a usable one.

    This used to fall back to `now()`. That invented a fact: 10 of 60 items carried a
    fabricated timestamp on 2026-08-19, all identical, and because the list sorts
    newest-first they occupied the entire top of the dashboard ahead of genuinely recent
    headlines — then re-dated themselves on the next fetch. Returning None keeps the
    "nothing here is generated" rule in the module docstring; `fetch_all` decides what an
    unknown date is worth.
    """
    if not raw:
        return None
    try:
        d = parsedate_to_datetime(raw)
        # Only reached for a *naive* datetime, where the feed omitted an offset.
        # Assuming UTC is the documented RSS fallback.
        return d if d.tzinfo else d.replace(tzinfo=UTC)
    except (TypeError, ValueError):
        try:
            # Python 3.11+ parses a trailing "Z" natively.
            d = datetime.fromisoformat(raw)
            return d if d.tzinfo else d.replace(tzinfo=UTC)
        except ValueError:
            return None


def load_seen(path=SEEN_PATH) -> dict[str, dict[str, str]]:
    """First-seen times for undated links. A missing or corrupt file is not fatal."""
    try:
        blob = json.loads(path.read_text())
        links = blob.get("links", {})
        return links if isinstance(links, dict) else {}
    except (OSError, ValueError):
        return {}


def save_seen(links: dict[str, dict[str, str]], path=SEEN_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = dict(sorted(links.items()))
    path.write_text(json.dumps({"schema_version": 1, "links": ordered}, indent=1))


def resolve_dates(items: list[Item], seen: dict[str, dict[str, str]],
                  now: datetime | None = None) -> dict[str, dict[str, str]]:
    """Give every undated item the time it was first fetched, and return the new store.

    Carrying the first-seen time forward is what stops an undated item re-dating itself
    on every fetch. It is still an estimate, not a publication time, so it is flagged as
    one rather than presented as the feed's own answer.
    """
    now = now or datetime.now(UTC)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    links = dict(seen)

    for it in items:
        if it.published_ts is not None:
            continue
        entry = links.get(it.link)
        if not isinstance(entry, dict) or "first_seen" not in entry:
            entry = {"first_seen": stamp}
        entry["last_seen"] = stamp
        links[it.link] = entry
        first = _parse_date(entry["first_seen"]) or now
        it.published = first.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        it.published_ts = first.timestamp()
        it.date_estimated = True

    # Forget links that have not appeared in any feed for a while, so the committed file
    # does not grow without bound over a season.
    cutoff = now - SEEN_RETENTION
    return {k: v for k, v in links.items()
            if (_parse_date(v.get("last_seen")) or now) >= cutoff}


def tag_teams(text: str) -> list[str]:
    low = text.lower()
    return [team for team, terms in TEAM_TERMS.items() if any(t in low for t in terms)]


def fetch_feed(source: str, url: str, timeout: int = 20) -> list[Item]:
    try:
        r = requests.get(url, headers=UA, timeout=timeout)
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except (requests.RequestException, ET.ParseError) as exc:
        print(f"  {source}: unavailable ({type(exc).__name__}) — skipping")
        return []

    items = []
    for node in root.iter("item"):
        title = _clean(node.findtext("title"), 200)
        link = (node.findtext("link") or "").strip()
        if not title or not link.startswith("http"):
            continue
        when = _parse_date(node.findtext("pubDate"))
        summary = _clean(node.findtext("description"))
        items.append(Item(
            title=title, link=link, source=source,
            # Left unset when the feed gave no usable date; fetch_all fills it in from
            # the first fetch that saw this link rather than inventing one here.
            published=when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if when else None,
            published_ts=when.timestamp() if when else None,
            summary=summary,
            teams=tag_teams(f"{title} {summary}"),
        ))
    print(f"  {source}: {len(items)} items")
    return items


def fetch_all(limit: int = 60, persist: bool = True) -> list[dict]:
    """All feeds, de-duplicated across sources, newest first."""
    fetched: list[Item] = []
    for source, url in FEEDS:
        fetched.extend(fetch_feed(source, url))

    # Before de-duplicating or sorting: both compare published_ts, which is None on an
    # item whose feed omitted a date until this resolves it.
    links = resolve_dates(fetched, load_seen())
    if persist:
        save_seen(links)
    estimated = sum(1 for i in fetched if i.date_estimated)
    if estimated:
        print(f"  {estimated} items had no usable pubDate — dated from first fetch")

    best: dict[str, Item] = {}
    for it in fetched:
        # Different outlets cover the same story; key on a normalised headline so
        # the section is not three copies of one race report.
        key = re.sub(r"[^a-z0-9]+", "", it.title.lower())[:60]
        if key not in best or it.published_ts > best[key].published_ts:
            best[key] = it
    items = sorted(best.values(), key=lambda i: i.published_ts, reverse=True)[:limit]
    # published_ts exists only to sort and de-duplicate; the client reads `published`.
    return [{k: v for k, v in asdict(i).items() if k != "published_ts"} for i in items]
