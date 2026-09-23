"""Turn a user's time expression into a UTC range.

The grammar is small and explicit. Anything outside it is rejected with a
reason rather than guessed at, so the agent asks the user instead of
investigating the wrong window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from ..config import format_iso, parse_iso

DAYPARTS: dict[str, tuple[int, int]] = {
    "morning": (6, 12),
    "afternoon": (12, 18),
    "evening": (18, 24),
    "night": (0, 6),
}

UNITS: dict[str, str] = {
    "minute": "minutes",
    "min": "minutes",
    "hour": "hours",
    "hr": "hours",
    "day": "days",
}

# The "T" separator is required, so that "2026-09-22 14:00 to 16:00" is read as a
# date plus a clock range rather than as one timestamp.
_ISO = r"\d{4}-\d{2}-\d{2}t\d{2}:\d{2}(?::\d{2})?(?:z|[+-]\d{2}:?\d{2})?"
_ISO_RE = re.compile(_ISO)
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_LAST_RE = re.compile(r"^last (\d{1,4}) (minutes?|mins?|hours?|hrs?|days?)$")
_CLOCK = r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"
_CLOCK_RANGE_RE = re.compile(rf"^{_CLOCK}\s*-\s*{_CLOCK}$")


class TimeResolutionError(Exception):
    """Raised when an expression cannot be turned into a usable range."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason  # "unresolvable" | "future_range"
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class ResolvedRange:
    start: datetime
    end: datetime
    assumptions: list[str]


def _normalize(expression: str) -> str:
    text = expression.strip().lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\b(between|from|this|the|during|for)\b", " ", text)
    text = re.sub(r"\b(to|and|until|till|through)\b", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def _day_start(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc)


def _clock_to_delta(hour: str, minute: str | None, meridiem: str | None) -> timedelta:
    h, m = int(hour), int(minute or 0)
    if meridiem:
        if not 1 <= h <= 12:
            raise TimeResolutionError("unresolvable", f"'{h}{meridiem}' is not a valid clock time.")
        h = 0 if h == 12 else h
        if meridiem == "pm":
            h += 12
    elif not 0 <= h <= 23:
        raise TimeResolutionError("unresolvable", f"'{hour}' is not a valid hour.")
    if not 0 <= m <= 59:
        raise TimeResolutionError("unresolvable", f"'{minute}' is not a valid minute.")
    return timedelta(hours=h, minutes=m)


def _parse_clock_range(text: str) -> tuple[timedelta, timedelta] | None:
    match = _CLOCK_RANGE_RE.match(text)
    if not match:
        return None
    h1, m1, mer1, h2, m2, mer2 = match.groups()
    # "2 - 4 pm": a single trailing meridiem applies to both ends.
    mer1 = mer1 or mer2
    return _clock_to_delta(h1, m1, mer1), _clock_to_delta(h2, m2, mer2)


def _span_for_remainder(remainder: str) -> tuple[timedelta, timedelta] | None:
    """Return the offsets within a day named by a daypart or a clock range."""
    if not remainder:
        return timedelta(0), timedelta(days=1)
    if remainder in DAYPARTS:
        start_h, end_h = DAYPARTS[remainder]
        return timedelta(hours=start_h), timedelta(hours=end_h)
    return _parse_clock_range(remainder)


def _most_recent_completed(span: tuple[timedelta, timedelta], now: datetime) -> tuple[datetime, datetime, str]:
    day = now.date()
    if _day_start(day) + span[1] > now:
        day -= timedelta(days=1)
    return _day_start(day) + span[0], _day_start(day) + span[1], day.isoformat()


def resolve(expression: str, now: datetime) -> ResolvedRange:
    """Resolve `expression` against `now`, or raise TimeResolutionError."""
    text = _normalize(expression)
    if not text:
        raise TimeResolutionError("unresolvable", "No time expression was given.")

    assumptions: list[str] = []
    iso_matches = _ISO_RE.findall(text)

    if len(iso_matches) >= 2:
        start, end = parse_iso(iso_matches[0]), parse_iso(iso_matches[1])
    elif len(iso_matches) == 1:
        raise TimeResolutionError(
            "unresolvable",
            "Only one timestamp was given. Provide both a start and an end time.",
        )
    elif last := _LAST_RE.match(text):
        amount = int(last.group(1))
        unit = UNITS[last.group(2).rstrip("s")]
        if amount == 0:
            raise TimeResolutionError("unresolvable", "The duration must be greater than zero.")
        start, end = now - timedelta(**{unit: amount}), now
    else:
        start, end = _resolve_calendar(text, now, assumptions)

    if start >= end:
        raise TimeResolutionError("unresolvable", "The start time is not before the end time.")
    if start >= now:
        raise TimeResolutionError(
            "future_range",
            f"That range starts at {format_iso(start)}, which is in the future. "
            f"The current time is {format_iso(now)}.",
        )
    if end > now:
        end = now
        assumptions.append(f"End time clipped to the current time, {format_iso(now)}.")
    return ResolvedRange(start=start, end=end, assumptions=assumptions)


def _resolve_calendar(text: str, now: datetime, assumptions: list[str]) -> tuple[datetime, datetime]:
    """Resolve the calendar forms: an optional date word plus an optional span."""
    day: date | None = None
    remainder = text

    if date_match := _DATE_RE.search(text):
        day = date.fromisoformat(date_match.group(1))
        remainder = _DATE_RE.sub(" ", text, count=1)
    else:
        for word, offset in (("today", 0), ("yesterday", -1), ("tomorrow", 1)):
            if re.search(rf"\b{word}\b", text):
                day = now.date() + timedelta(days=offset)
                remainder = re.sub(rf"\b{word}\b", " ", text, count=1)
                break

    remainder = re.sub(r"\s+", " ", remainder).strip()
    span = _span_for_remainder(remainder)
    if span is None:
        raise TimeResolutionError(
            "unresolvable",
            f"'{text}' is not a supported time expression. Supported forms: "
            "today/yesterday (optionally with morning, afternoon, evening or night), "
            "last N minutes/hours/days, a clock range such as '2 PM to 4 PM', "
            "or two ISO 8601 timestamps.",
        )

    if day is None:
        start, end, assumed = _most_recent_completed(span, now)
        assumptions.append(f"Date not given; assumed {assumed}, the most recent completed occurrence.")
        return start, end
    return _day_start(day) + span[0], _day_start(day) + span[1]


def validate_range(start: datetime, end: datetime, now: datetime, max_window_days: int) -> list[str]:
    """Check a range that is about to be passed to a data tool."""
    errors: list[str] = []
    if start >= end:
        errors.append(f"start_time ({format_iso(start)}) must be before end_time ({format_iso(end)}).")
    if end > now:
        errors.append(f"end_time ({format_iso(end)}) is in the future. The current time is {format_iso(now)}.")
    if end - start > timedelta(days=max_window_days):
        errors.append(f"The window must be at most {max_window_days} days.")
    return errors
