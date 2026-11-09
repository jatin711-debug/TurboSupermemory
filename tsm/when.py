"""Dates: when a memory was said, and the days a question points at.

``said_on`` reads the calendar date out of a stored timestamp.

``question_windows`` resolves the time expressions in a question ("ten days
ago", "last weekend", "in the past two months", "in March") against the date
it is asked, into stretches of days. ``Memory.recall(now=...)`` matches those
against the dates memories were said. It is rule-based (English only) and
leaves what it does not recognise alone: an expression it cannot place gets
no window rather than a guess. Nothing here calls a model or an engine.

Until 2026-11-09 this module also wrote the date a relative time points to
after it in a memory's text (``annotate``, ``recall(event_dates=True)``:
"two weeks ago [week of 2023-05-08]"). Judged over 470 questions that came
out level (8 gained, 4 lost) and it was removed
(``benchmarks/PHASE_PROGRESS.md``, 2026-11-09).
"""

import math
import re
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Tuple

_MONTHS = {m: i for i, m in enumerate(
    ("january february march april may june july august september october november "
     "december").split(), 1)}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})
_MONTHS["sept"] = 9
_WEEKDAYS = {d: i for i, d in enumerate(
    "monday tuesday wednesday thursday friday saturday sunday".split())}
_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
            "twelve": 12, "a couple of": 2, "couple of": 2}

_DATE_STAMP = re.compile(r"\s*(\d{4})[-/](\d{2})[-/](\d{2})")

_MONTH_NAME = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|" \
              r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_WEEKDAY = r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
_COUNT = r"(\d{1,3}|a couple of|couple of|an?|one|two|three|four|five|six|seven|eight|nine|ten|" \
         r"eleven|twelve)"
_UNIT = r"(day|week|month|year)s?"

# Most specific first; the text a match covers is not matched again.
_PATTERNS = [
    ("iso", re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")),
    ("month_day", re.compile(
        rf"\b{_MONTH_NAME}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}})\b)?", re.I)),
    ("day_month", re.compile(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH_NAME}\b(?:,?\s+(\d{{4}})\b)?", re.I)),
    ("month_year", re.compile(rf"\b{_MONTH_NAME}\s+(?:of\s+)?(\d{{4}})\b", re.I)),
    ("nth_of_month", re.compile(
        r"\b(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)\s+of\s+(last|this|next)\s+month\b", re.I)),
    ("ago", re.compile(rf"\b{_COUNT}\s+{_UNIT}\s+ago\b", re.I)),
    ("last_weekday", re.compile(rf"\b(last|this past|next)\s+{_WEEKDAY}\b", re.I)),
    ("relative_unit", re.compile(r"\b(last|this|next)\s+(week|weekend|month|year)\b", re.I)),
    ("relative_day", re.compile(r"\b(the day before yesterday|yesterday|tomorrow)\b", re.I)),
    ("in_month", re.compile(rf"\b(?:in|during|since|until|by|back in|early|late)\s+{_MONTH_NAME}\b"
                            r"(?!\s+\d)", re.I)),
]
# Not matched: "in two weeks" (as often a duration as a date), "today",
# "this morning", a bare year.


def said_on(timestamp) -> Optional[date]:
    """The calendar date of a stored timestamp: an ISO-like string
    (``2023-05-22``, ``2023/05/22 (Mon) 14:05``) or epoch seconds or
    milliseconds; ``None`` when there is no date in it."""
    if isinstance(timestamp, str):
        match = _DATE_STAMP.match(timestamp)
        if match:
            try:
                return date(*(int(g) for g in match.groups()))
            except ValueError:
                return None
        try:
            timestamp = float(timestamp)
        except ValueError:
            return None
    if (isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool)
            and math.isfinite(timestamp) and timestamp >= 1e8):
        seconds = timestamp / 1000.0 if timestamp >= 1e11 else float(timestamp)
        try:
            return datetime.fromtimestamp(seconds, timezone.utc).date()
        except (OverflowError, OSError, ValueError):
            return None
    return None


def _count(word: str) -> Optional[int]:
    word = word.lower()
    if word.isdigit():
        return int(word)
    return _NUMBERS.get(word)


def _month(word: str) -> int:
    return _MONTHS[word.lower().rstrip(".")[:4] if word.lower().startswith("sept")
                   else word.lower().rstrip(".")[:3]]


def _add_months(day: date, months: int) -> date:
    index = day.year * 12 + day.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


# A question can point at a stretch of time rather than at a topic: "what did
# I buy ten days ago", "the trips I took in the past three months". The
# patterns above say when; this one also covers "in the past N weeks".
_PAST_SPAN = re.compile(
    rf"\b(?:in|over|during|within|for)\s+the\s+(?:past|last)\s+(?:{_COUNT}\s+|(few)\s+)?"
    rf"{_UNIT}\b", re.I)
_WINDOW_PATTERNS = [("past_span", _PAST_SPAN)] + _PATTERNS


def _shift_months(day: date, months: int) -> date:
    """``day`` moved by whole months, its day of the month kept where the
    month has it (else the month's last day)."""
    first = _add_months(day, months)
    last = (_add_months(first, 1) - timedelta(days=1)).day
    return first.replace(day=min(day.day, last))


def _month_span(first: date) -> Tuple[date, date]:
    return first, _add_months(first, 1) - timedelta(days=1)


def _latest_month(month: int, now: date, year: Optional[str] = None) -> date:
    """The first of ``month``: in ``year`` when given, else the latest such
    month that has begun by ``now``."""
    if year:
        return date(int(year), month, 1)
    return date(now.year if month <= now.month else now.year - 1, month, 1)


def _occurred(kind: str, groups: Tuple[str, ...], now: date) -> Optional[Tuple[date, date]]:
    """The days an expression in a question asked on ``now`` points to, first
    and last; ``None`` for one that points ahead or is too vague to bound."""
    if kind == "past_span":
        count, few, unit = groups
        n = 3 if few else (_count(count) if count else 1)
        if n is None or n > 100:
            return None
        unit = unit.lower()
        if unit == "day":
            return now - timedelta(days=n), now
        if unit == "week":
            return now - timedelta(weeks=n), now
        if unit == "month":
            return _shift_months(now, -n), now
        return _shift_months(now, -12 * n), now
    if kind == "ago":
        n = _count(groups[0])
        if n is None or n > 100:
            return None
        unit = groups[1].lower()
        if unit == "day":
            center, slack = now - timedelta(days=n), max(1, n // 5)
        elif unit == "week":
            center, slack = now - timedelta(weeks=n), max(3, 7 * n // 5)
        elif unit == "month":
            center, slack = _shift_months(now, -n), 15
        else:
            center, slack = _shift_months(now, -12 * n), 60
        return center - timedelta(days=slack), center + timedelta(days=slack)
    if kind == "relative_day":
        back = {"the day before yesterday": 2, "yesterday": 1, "tomorrow": -1}[groups[0].lower()]
        if back < 0:
            return None
        day = now - timedelta(days=back)
        return day, day
    if kind == "last_weekday":
        if groups[0].lower() == "next":
            return None
        day = now - timedelta(days=(now.weekday() - _WEEKDAYS[groups[1].lower()]) % 7 or 7)
        return day, day
    if kind == "relative_unit":
        which, unit = groups[0].lower(), groups[1].lower()
        if which == "next" or unit == "year":
            return None
        if unit == "week":
            monday = now - timedelta(days=now.weekday())
            if which == "this":
                return monday, now
            return monday - timedelta(weeks=1), monday - timedelta(days=1)
        if unit == "weekend":
            if which == "this" and now.weekday() < 5:
                return None  # still to come
            saturday = now - timedelta(days=(now.weekday() - 5) % 7)
            if which == "last" and now.weekday() >= 5:
                saturday -= timedelta(weeks=1)
            return saturday, saturday + timedelta(days=1)
        first = date(now.year, now.month, 1)
        if which == "this":
            return first, now
        return _month_span(_add_months(first, -1))
    if kind == "nth_of_month":
        if groups[1].lower() == "next":
            return None
        first = _add_months(now, {"last": -1, "this": 0}[groups[1].lower()])
        try:
            day = first.replace(day=int(groups[0]))
        except ValueError:
            return None
        return day, day
    if kind == "iso":
        try:
            day = date(int(groups[0]), int(groups[1]), int(groups[2]))
        except ValueError:
            return None
        return day, day
    if kind in ("month_day", "day_month"):
        month_word, day_word, year = ((groups[0], groups[1], groups[2]) if kind == "month_day"
                                      else (groups[1], groups[0], groups[2]))
        first = _latest_month(_month(month_word), now, year)
        try:
            day = first.replace(day=int(day_word))
        except ValueError:
            return None
        if not year and day > now:
            try:
                day = day.replace(year=day.year - 1)
            except ValueError:
                return None
        return day, day
    if kind in ("month_year", "in_month"):
        year = groups[1] if kind == "month_year" else None
        return _month_span(_latest_month(_month(groups[0]), now, year))
    return None


def question_windows(text: str, now) -> List[Tuple[date, date]]:
    """The stretches of days a question asked on ``now`` (a date, a datetime,
    or a timestamp ``said_on`` reads) points to, first and last day of each,
    in the order they appear: "ten days ago" asked on 2023-03-25 gives about
    2023-03-15, "last week" the Monday to Sunday before, "in the past two
    months" everything since 2023-01-25. Empty when it points to no time
    before ``now``, or when ``now`` is not known.

    A day here is when something happened. A memory is dated by when it was
    said, which is that day or later ("I got it yesterday"), so a reader of
    these windows widens them forward (``Memory.recall``).
    """
    if isinstance(now, datetime):
        now = now.date()
    elif not isinstance(now, date):
        now = said_on(now)
    if now is None or not text:
        return []
    found: List[Tuple[int, Tuple[date, date]]] = []
    taken: List[Tuple[int, int]] = []
    for kind, pattern in _WINDOW_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < b and a < end for a, b in taken):
                continue
            taken.append((start, end))
            span = _occurred(kind, match.groups(), now)
            if span and span[0] <= now:
                found.append((start, (span[0], min(span[1], now))))
    return [span for _, span in sorted(found)]
