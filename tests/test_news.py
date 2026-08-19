"""Tests for the news layer.

Feed content is third-party input, so the tests that matter here are the ones about
not trusting it: markup must be stripped, lengths bounded, and malformed dates must
not take the build down.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from apex.news import (
    Item,
    _clean,
    _parse_date,
    load_seen,
    resolve_dates,
    save_seen,
    tag_teams,
)


def test_clean_strips_markup_and_entities():
    raw = "<p>Alonso &amp; Stroll <a href='http://x'>react</a></p>"
    assert _clean(raw) == "Alonso & Stroll react"


def test_clean_removes_script_content_markup():
    """Any markup is reduced to text; nothing survives that a renderer could execute."""
    out = _clean("<script>alert('x')</script>Hamilton fastest")
    assert "<" not in out and ">" not in out
    assert "Hamilton fastest" in out


def test_clean_truncates_with_ellipsis():
    out = _clean("x" * 500, limit=50)
    assert len(out) <= 51 and out.endswith("…")


def test_clean_handles_missing_text():
    assert _clean(None) == ""
    assert _clean("") == ""


def test_parse_date_accepts_rfc822_and_iso():
    assert _parse_date("Fri, 24 Jul 2026 18:03:56 GMT").year == 2026
    assert _parse_date("2026-07-24T18:03:56Z").year == 2026


def test_parse_date_returns_none_when_the_feed_gives_nothing_usable():
    """A broken pubDate must not crash the build, and must not be invented either."""
    assert _parse_date("not a date") is None
    assert _parse_date(None) is None
    assert _parse_date("") is None


def test_tag_teams_matches_constructors_and_drivers():
    assert "Mercedes" in tag_teams("Antonelli takes pole in Hungary")
    assert "Ferrari" in tag_teams("Hamilton leads Leclerc in practice")
    assert "Cadillac" in tag_teams("Bottas reflects on debut season")


def test_tag_teams_is_case_insensitive_and_can_be_empty():
    assert "Red Bull Racing" in tag_teams("VERSTAPPEN on the front row")
    assert tag_teams("FIA announces calendar changes") == []


def test_tag_teams_can_return_several():
    tags = tag_teams("Hamilton beats Verstappen to victory")
    assert "Ferrari" in tags and "Red Bull Racing" in tags


# --- undated items -----------------------------------------------------------------
# The bug these pin: `_parse_date` used to return now() for an item whose feed omitted a
# pubDate. Ten of sixty items carried a fabricated timestamp, all identical, and since
# the list sorts newest-first they took the whole top of the dashboard — then re-dated
# themselves on the next fetch, so the refresh could not tell a real change from a tick.

def _undated(link="http://x/1", title="Undated headline"):
    return Item(title=title, link=link, source="Test", published=None,
                published_ts=None, summary="", teams=[])


def _dated(link="http://x/2", title="Dated headline"):
    when = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    return Item(title=title, link=link, source="Test",
                published="2026-08-01T12:00:00Z", published_ts=when.timestamp(),
                summary="", teams=[])


def test_undated_item_is_dated_from_first_fetch_and_flagged():
    it = _undated()
    now = datetime(2026, 8, 19, 9, 0, tzinfo=UTC)
    links = resolve_dates([it], {}, now=now)

    assert it.published == "2026-08-19T09:00:00Z"
    assert it.date_estimated is True
    assert links[it.link]["first_seen"] == "2026-08-19T09:00:00Z"


def test_undated_item_keeps_its_original_date_on_a_later_fetch():
    """The whole point: re-fetching must not make an old item look new again."""
    first = datetime(2026, 8, 19, 9, 0, tzinfo=UTC)
    links = resolve_dates([_undated()], {}, now=first)

    later = _undated()
    links = resolve_dates([later], links, now=first + timedelta(days=3))

    assert later.published == "2026-08-19T09:00:00Z"
    assert links[later.link]["first_seen"] == "2026-08-19T09:00:00Z"
    assert links[later.link]["last_seen"] == "2026-08-22T09:00:00Z"


def test_a_real_pubdate_is_never_overwritten_or_flagged():
    it = _dated()
    links = resolve_dates([it], {}, now=datetime(2026, 8, 19, 9, 0, tzinfo=UTC))

    assert it.published == "2026-08-01T12:00:00Z"
    assert it.date_estimated is False
    assert links == {}          # only undated links are recorded


def test_dated_items_sort_above_an_older_undated_one():
    now = datetime(2026, 8, 19, 9, 0, tzinfo=UTC)
    old = _undated()
    resolve_dates([old], {}, now=now - timedelta(days=10))
    fresh = _dated()
    fresh.published_ts = (now - timedelta(hours=1)).timestamp()

    order = sorted([old, fresh], key=lambda i: i.published_ts, reverse=True)
    assert order[0] is fresh


def test_store_forgets_links_that_stopped_appearing():
    now = datetime(2026, 8, 19, 9, 0, tzinfo=UTC)
    stale = {"http://x/gone": {"first_seen": "2026-01-01T00:00:00Z",
                               "last_seen": "2026-06-01T00:00:00Z"}}
    recent = {"http://x/here": {"first_seen": "2026-08-01T00:00:00Z",
                               "last_seen": "2026-08-18T00:00:00Z"}}

    links = resolve_dates([], {**stale, **recent}, now=now)

    assert "http://x/gone" not in links
    assert "http://x/here" in links


def test_store_roundtrips_and_survives_a_corrupt_file(tmp_path):
    path = tmp_path / "news_seen.json"
    assert load_seen(path) == {}                      # missing file is not fatal

    links = {"http://x/1": {"first_seen": "2026-08-19T09:00:00Z",
                            "last_seen": "2026-08-19T09:00:00Z"}}
    save_seen(links, path)
    assert load_seen(path) == links

    path.write_text("{not json")
    assert load_seen(path) == {}
