"""When a memory says something happened, as opposed to when it was said.

A message dated 2023-05-22 that says "I saw the dentist two weeks ago" is
about the week of 2023-05-08. A reader told only the first date has to do
that arithmetic itself, and readers get it wrong: of the temporal questions
answered wrong with all their evidence present, most needed a date computed
from a relative one. Systems that store event time apart from mention time
(Zep's bi-temporal edges, Chronos's dated event records, Hindsight's
occurrence intervals) report their largest gains on exactly these questions.

``annotate`` resolves the time expressions in a text against the date it was
said and writes the date each one points to after it, in brackets:

    annotate("I saw the dentist two weeks ago", date(2023, 5, 22))
    -> "I saw the dentist two weeks ago [week of 2023-05-08]"

It is rule-based (English only), runs at read time, and leaves what it does
not recognise alone: an expression it cannot place precisely gets no
annotation rather than a guess. ``said_on`` reads the date out of a stored
timestamp. Nothing here calls a model or an engine.
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

# Most specific first; each match is annotated once and the text it covers
# is not matched again.
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
# Not annotated: "in two weeks" (as often a duration as a date), "today" and
# "this morning" (the date the message was said, which its tag already
# gives), a bare year (already a date).


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


def _nearest_year(month: int, day: int, said: date) -> Optional[date]:
    """The date with this month and day nearest to ``said``."""
    best = None
    for year in (said.year - 1, said.year, said.year + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if best is None or abs((candidate - said).days) < abs((best - said).days):
            best = candidate
    return best


def _week_of(day: date) -> str:
    return f"week of {day - timedelta(days=day.weekday()):%Y-%m-%d}"


def _resolve(kind: str, groups: Tuple[str, ...], said: date) -> Optional[str]:
    """The date an expression points to, as written in its annotation."""
    if kind == "iso":
        return None  # already a date
    if kind in ("month_day", "day_month"):
        month_word, day_word, year = ((groups[0], groups[1], groups[2]) if kind == "month_day"
                                      else (groups[1], groups[0], groups[2]))
        if year:
            return None  # already a full date
        resolved = _nearest_year(_month(month_word), int(day_word), said)
        return f"{resolved:%Y-%m-%d}" if resolved else None
    if kind == "month_year":
        return None  # already as precise as it gets
    if kind == "nth_of_month":
        first = _add_months(said, {"last": -1, "this": 0, "next": 1}[groups[1].lower()])
        try:
            return f"{first.replace(day=int(groups[0])):%Y-%m-%d}"
        except ValueError:
            return None
    if kind == "ago":
        n = _count(groups[0])
        if n is None or n > 100:
            return None
        unit = groups[1].lower()
        if unit == "day":
            return f"{said - timedelta(days=n):%Y-%m-%d}"
        if unit == "week":
            return _week_of(said - timedelta(weeks=n))
        if unit == "month":
            return f"{_add_months(said, -n):%Y-%m}"
        return f"{said.year - n}"
    if kind == "last_weekday":
        which, weekday = groups[0].lower(), _WEEKDAYS[groups[1].lower()]
        if which == "next":
            ahead = (weekday - said.weekday()) % 7 or 7
            return f"{said + timedelta(days=ahead):%Y-%m-%d}"
        back = (said.weekday() - weekday) % 7 or 7
        return f"{said - timedelta(days=back):%Y-%m-%d}"
    if kind == "relative_unit":
        which, unit = groups[0].lower(), groups[1].lower()
        step = {"last": -1, "this": 0, "next": 1}[which]
        if unit == "week":
            return _week_of(said + timedelta(weeks=step))
        if unit == "weekend":
            saturday = said + timedelta(days=(5 - said.weekday()) % 7)
            if which == "last" or (which == "this" and said.weekday() == 6):
                saturday = said - timedelta(days=(said.weekday() - 5) % 7 or 7)
            elif which == "next":
                saturday += timedelta(weeks=1)
            return f"weekend of {saturday:%Y-%m-%d}"
        if unit == "month":
            return f"{_add_months(said, step):%Y-%m}"
        return f"{said.year + step}"
    if kind == "relative_day":
        offset = {"the day before yesterday": -2, "yesterday": -1, "tomorrow": 1}[
            groups[0].lower()]
        return f"{said + timedelta(days=offset):%Y-%m-%d}"
    if kind == "in_month":
        month = _month(groups[0])
        # The nearest such month, the past one on a tie: "in August" said in
        # May is a plan, said in December it is history.
        ahead = (month - said.month) % 12
        year = said.year + (said.month + ahead > 12) if ahead < 6 else (
            said.year - (month > said.month))
        return f"{year}-{month:02d}"
    return None


def expressions(text: str, said: date) -> List[Tuple[int, int, str]]:
    """``(start, end, resolved)`` for each time expression in ``text`` that
    can be placed relative to ``said``, in order."""
    found: List[Tuple[int, int, str]] = []
    taken: List[Tuple[int, int]] = []
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(text or ""):
            start, end = match.span()
            if any(start < b and a < end for a, b in taken):
                continue
            taken.append((start, end))
            resolved = _resolve(kind, match.groups(), said)
            if resolved:
                found.append((start, end, resolved))
    return sorted(found)


def annotate(text: str, said: Optional[date]) -> str:
    """``text`` with the date each of its relative time expressions points
    to written after it in brackets. Unchanged without ``said``."""
    if not text or said is None:
        return text
    found = expressions(text, said)
    if not found:
        return text
    out, last = [], 0
    for start, end, resolved in found:
        out.append(text[last:end])
        out.append(f" [{resolved}]")
        last = end
    out.append(text[last:])
    return "".join(out)
