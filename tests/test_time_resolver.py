"""The time resolver: the supported grammar, and what it refuses."""

from __future__ import annotations

import pytest

from incident_agent.config import format_iso
from incident_agent.tools.time_resolver import TimeResolutionError, resolve, validate_range
from tests.conftest import NOW


@pytest.mark.parametrize(
    "expression,start,end",
    [
        ("yesterday", "2026-09-22T00:00:00Z", "2026-09-23T00:00:00Z"),
        ("yesterday afternoon", "2026-09-22T12:00:00Z", "2026-09-22T18:00:00Z"),
        ("yesterday morning", "2026-09-22T06:00:00Z", "2026-09-22T12:00:00Z"),
        ("yesterday evening", "2026-09-22T18:00:00Z", "2026-09-23T00:00:00Z"),
        ("yesterday night", "2026-09-22T00:00:00Z", "2026-09-22T06:00:00Z"),
        ("yesterday between 2 PM and 4 PM", "2026-09-22T14:00:00Z", "2026-09-22T16:00:00Z"),
        ("2026-09-22 from 14:00 to 16:00", "2026-09-22T14:00:00Z", "2026-09-22T16:00:00Z"),
        ("2026-09-21", "2026-09-21T00:00:00Z", "2026-09-22T00:00:00Z"),
        ("last 2 hours", "2026-09-23T08:00:00Z", "2026-09-23T10:00:00Z"),
        ("last 90 minutes", "2026-09-23T08:30:00Z", "2026-09-23T10:00:00Z"),
        ("last 3 days", "2026-09-20T10:00:00Z", "2026-09-23T10:00:00Z"),
        ("2026-09-22T14:00:00Z to 2026-09-22T16:00:00Z", "2026-09-22T14:00:00Z", "2026-09-22T16:00:00Z"),
    ],
)
def test_supported_expressions(expression, start, end):
    resolved = resolve(expression, NOW)
    assert (format_iso(resolved.start), format_iso(resolved.end)) == (start, end)


@pytest.mark.parametrize("expression", ["2 PM to 4 PM", "14:00-16:00", "afternoon"])
def test_missing_date_uses_most_recent_completed_occurrence_and_says_so(expression):
    resolved = resolve(expression, NOW)
    assert resolved.start.date().isoformat() == "2026-09-22"
    assert any("2026-09-22" in a for a in resolved.assumptions)


def test_today_is_clipped_to_now_and_the_clip_is_reported():
    resolved = resolve("today", NOW)
    assert format_iso(resolved.end) == "2026-09-23T10:00:00Z"
    assert any("clipped" in a for a in resolved.assumptions)


@pytest.mark.parametrize(
    "expression,reason",
    [
        ("tomorrow afternoon", "future_range"),
        ("today afternoon", "future_range"),
        ("recently", "unresolvable"),
        ("during the outage", "unresolvable"),
        ("", "unresolvable"),
        ("yesterday 4 PM to 2 PM", "unresolvable"),
        ("2026-09-22T14:00:00Z", "unresolvable"),
        ("last 0 hours", "unresolvable"),
        ("yesterday 25:00 to 26:00", "unresolvable"),
    ],
)
def test_refused_expressions(expression, reason):
    with pytest.raises(TimeResolutionError) as caught:
        resolve(expression, NOW)
    assert caught.value.reason == reason


def test_now_is_injected_not_read_from_the_system_clock():
    earlier = resolve("yesterday", NOW.replace(year=2025))
    assert earlier.start.year == 2025


def test_validate_range_rejects_reversed_future_and_oversized_windows():
    start, end = resolve("yesterday", NOW).start, resolve("yesterday", NOW).end
    assert validate_range(start, end, NOW, 7) == []
    assert validate_range(end, start, NOW, 7)  # reversed
    assert validate_range(start, NOW.replace(day=30), NOW, 7)  # in the future
    assert validate_range(start.replace(month=8, day=1), end, NOW, 7)  # too long
