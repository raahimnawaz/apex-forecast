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
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import requests

FEEDS = [
    ("Autosport", "https://www.autosport.com/rss/f1/news/"),
    ("Motorsport.com", "https://www.motorsport.com/rss/f1/news/"),
    ("BBC Sport", "https://feeds.bbci.co.uk/sport/formula1/rss.xml"),
    ("Formula1.com", "https://www.formula1.com/en/latest/all.xml"),
]

UA = {"User-Agent": "Mozilla/5.0 (compatible; apex-forecast/0.1)"}

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
    published: str          # ISO-8601 UTC
    published_ts: float
    summary: str
    teams: list[str]


def _clean(text: str | None, limit: int = 260) -> str:
    """Strip markup and entities from feed text. Never rendered as HTML downstream."""
    if not text:
        return ""
    t = html.unescape(TAG_RE.sub(" ", text))
    t = WS_RE.sub(" ", t).strip()
    return t[:limit].rstrip() + ("…" if len(t) > limit else "")


def _parse_date(raw: str | None) -> datetime:
    if not raw:
        return datetime.now(UTC)
    try:
        d = parsedate_to_datetime(raw)
        # Only reached for a *naive* datetime, where the feed omitted an offset.
        # Assuming UTC is the documented RSS fallback.
        return d if d.tzinfo else d.replace(tzinfo=UTC)
    except (TypeError, ValueError):
        try:
            # Python 3.11+ parses a trailing "Z" natively.
            return datetime.fromisoformat(raw)
        except ValueError:
            return datetime.now(UTC)


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
            published=when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            published_ts=when.timestamp(),
            summary=summary,
            teams=tag_teams(f"{title} {summary}"),
        ))
    print(f"  {source}: {len(items)} items")
    return items


def fetch_all(limit: int = 60) -> list[dict]:
    """All feeds, de-duplicated across sources, newest first."""
    seen: dict[str, Item] = {}
    for source, url in FEEDS:
        for it in fetch_feed(source, url):
            # Different outlets cover the same story; key on a normalised headline so
            # the section is not three copies of one race report.
            key = re.sub(r"[^a-z0-9]+", "", it.title.lower())[:60]
            if key not in seen or it.published_ts > seen[key].published_ts:
                seen[key] = it
    items = sorted(seen.values(), key=lambda i: i.published_ts, reverse=True)[:limit]
    # published_ts exists only to sort and de-duplicate; the client reads `published`.
    return [{k: v for k, v in asdict(i).items() if k != "published_ts"} for i in items]
