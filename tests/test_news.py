"""Tests for the news layer.

Feed content is third-party input, so the tests that matter here are the ones about
not trusting it: markup must be stripped, lengths bounded, and malformed dates must
not take the build down.
"""

from __future__ import annotations

from apex.news import _clean, _parse_date, tag_teams


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


def test_parse_date_survives_garbage():
    """A broken pubDate must not crash the build — it falls back to 'now'."""
    assert _parse_date("not a date").year >= 2026
    assert _parse_date(None).year >= 2026


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
