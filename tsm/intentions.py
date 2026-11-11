"""Prospective memory: what the user wants done later, and when it is due.

A store that only answers questions gets intentions wrong in a specific way:
a reminder surfaces when something similar is asked about, not when it is
due. On ProactiveBench plain retrieval scored *below* no memory (0.364
against 0.510) because relevance was mistaken for readiness. So an intention
here is a record with a lifecycle, kept apart from the memories and judged
for readiness separately from retrieval:

- **formation** (``Memory.add``): a user message with a reminder cue
  ("remind me to ...", "don't let me forget ...") or a commitment with a
  future time in it ("I need to renew my passport by March 3") becomes an
  intention: its objective (the sentence), when it is due (``future_window``
  resolves "tomorrow", "next Friday", "in two weeks", "by March 3" against
  the message's date), and links to the memories of the message it came from
  (``source_ids``);
- **maintenance** (``Memory.consolidate``, never on a timer): a later
  message that says it is done or no longer wanted ("I already booked it",
  "never mind about the passport") closes the nearest pending intention, and
  a new intention that restates a pending one ("move the dentist to Friday")
  replaces it (``superseded_by``). Both are written as payload patches in one
  batch per pass;
- **readiness** (``Memory.due``): pending intentions whose time has come, plus
  those the caller's current context brings to mind, judged ``act`` (time
  has come) or ``observe`` (only the context matches). A ``readiness_judge``
  can re-judge at most ``top_k`` of them as act / wait / observe / satisfied
  / invalidated. ``due`` is a read: it changes nothing, the caller closes an
  intention with ``Memory.resolve_intention``.

An intention persists until something reconsiders it: a completion, a
replacement, or the caller. Nothing expires on its own.

Intentions live in their own engine (``<db_path>/tsm_intentions``) with the
same embedder, one scope per user. The memories' engine is not touched, so
enabling them changes no recall result: an engine's word index and concept
graph are shared by its scopes, and derived records kept beside the memories
once changed 17 of 20 unrelated contexts.

Status (2026-11-11): a prototype with unit tests, not yet measured. The cue
rules are English and rule-based; the cosine floors for completion and
replacement (``COMPLETION_MIN_COSINE``, ``RESCHEDULE_MIN_COSINE``) belong to an
embedder's geometry and were not tuned on any. Its evaluation (PM-Bench,
ProactiveBench; kill it if it does not beat a plain to-do ledger) is Phase 3
of ``reports/Cognitive architecture for TurboSuperMemory.md``.
"""

import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .when import _WEEKDAYS, _add_months, _count, _month, _month_span, as_date

# A message that asks to be reminded, whatever its tense.
REMINDER_CUES = re.compile(
    r"\b(?:remind me|don'?t let me forget|do not let me forget|make sure i|note to self|"
    r"add (?:it |this |that )?to my (?:to-?do|todo|task) list|"
    r"put (?:it |this |that )?(?:on|in) my (?:calendar|list|agenda)|i need a reminder)\b", re.I)
# A commitment the speaker makes; an intention only with a future time.
COMMITMENT_CUES = re.compile(
    r"\b(?:i|we)(?:\s+(?:need|have|got|plan|want|intend|promised|hope)\s+to|"
    r"\s+(?:must|should|will|shall|gotta)|'ll|\s+am going to|'m going to|"
    r"\s+are going to|'re going to)\b", re.I)
# A message that says an intention was carried out ...
DONE_CUES = re.compile(
    r"\b(?:(?:i|we)(?:'ve| have)?\s+(?:already|just|finally)\s+\w+|"
    r"(?:it'?s|that'?s|this is)\s+(?:done|taken care of|sorted|handled)|"
    r"(?:i|we)\s+(?:did|finished|completed|booked|called|paid|sent|bought|"
    r"submitted|renewed|returned|picked up)\s+(?:it|that|them))\b", re.I)
# ... or is no longer wanted.
CANCEL_CUES = re.compile(
    r"\b(?:no need to remind me|cancel (?:the|that|my|this) reminder|never ?mind|"
    r"forget (?:about )?(?:it|that|the|my)|(?:i|we) (?:don'?t|no longer) need to|"
    r"not (?:going|planning) to .{0,40} anymore|(?:it'?s|that'?s) (?:been )?(?:cancell?ed|called off))\b",
    re.I)

# Cosine floors, for ``maintain``. Embedder-dependent and untuned (see above).
COMPLETION_MIN_COSINE = 0.55
RESCHEDULE_MIN_COSINE = 0.80
# ``due``: how close the context must be to a pending intention to bring it
# to mind without its time having come.
CUE_MIN_COSINE = 0.50

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_MONTH = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|" \
         r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_WEEKDAY = r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
_COUNT = r"(\d{1,3}|a couple of|couple of|an?|one|two|three|four|five|six|seven|eight|nine|ten|" \
         r"eleven|twelve)"

# Most specific first; the text a match covers is not matched again.
_FUTURE = [
    ("iso", re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")),
    ("month_day", re.compile(
        rf"\b(by|before|on|until|till)?\s*{_MONTH}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b"
        r"(?:,?\s+(\d{4})\b)?", re.I)),
    ("day_month", re.compile(
        rf"\b(by|before|on|until|till)?\s*(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?"
        rf"{_MONTH}\b(?:,?\s+(\d{{4}})\b)?", re.I)),
    ("after_tomorrow", re.compile(r"\bthe day after tomorrow\b", re.I)),
    ("tomorrow", re.compile(r"\b(by\s+)?tomorrow\b", re.I)),
    ("today", re.compile(r"\b(today|tonight|this (?:morning|afternoon|evening))\b", re.I)),
    ("in_count", re.compile(rf"\bin\s+{_COUNT}\s+(day|week|month|year)s?\b", re.I)),
    ("end_of", re.compile(r"\b(?:by\s+)?(?:the\s+)?end of (?:the|this|next)\s+(week|month|year)\b",
                          re.I)),
    ("weekday", re.compile(rf"\b(by|before|on|this|next|coming)\s+{_WEEKDAY}\b", re.I)),
    ("next_unit", re.compile(r"\b(this|next|coming)\s+(week|weekend|month|year)\b", re.I)),
    ("in_month", re.compile(rf"\b(?:in|by|during|before)\s+{_MONTH}\b(?!\s+\d)", re.I)),
]


def _resolve(kind: str, groups: Tuple, today: date) -> Optional[Tuple[date, date]]:
    """The days an expression said on ``today`` points ahead to, first and
    last; ``None`` for one that points back or cannot be placed. A deadline
    ("by", "before", "until") runs from ``today`` to its date."""
    deadline = False
    if kind == "iso":
        try:
            day = date(int(groups[0]), int(groups[1]), int(groups[2]))
        except ValueError:
            return None
        return (day, day) if day >= today else None
    if kind in ("month_day", "day_month"):
        lead, a, b, year = groups
        month_word, day_word = (a, b) if kind == "month_day" else (b, a)
        deadline = bool(lead) and lead.lower() in ("by", "before", "until", "till")
        try:
            day = date(int(year) if year else today.year, _month(month_word), int(day_word))
        except ValueError:
            return None
        if day < today:
            if year:
                return None
            try:
                day = day.replace(year=day.year + 1)
            except ValueError:
                return None
        return (today, day) if deadline else (day, day)
    if kind == "after_tomorrow":
        day = today + timedelta(days=2)
        return day, day
    if kind == "tomorrow":
        day = today + timedelta(days=1)
        return (today, day) if groups[0] else (day, day)
    if kind == "today":
        return today, today
    if kind == "in_count":
        n = _count(groups[0])
        if n is None or n > 100:
            return None
        unit = groups[1].lower()
        if unit == "day":
            day = today + timedelta(days=n)
        elif unit == "week":
            day = today + timedelta(weeks=n)
        elif unit == "month":
            day = _add_months(today, n).replace(day=1)
            day = day.replace(day=min(today.day, _month_span(day)[1].day))
        else:
            try:
                day = today.replace(year=today.year + n)
            except ValueError:
                day = today.replace(year=today.year + n, day=28)
        return day, day
    if kind == "end_of":
        unit = groups[0].lower()
        if unit == "week":
            day = today + timedelta(days=6 - today.weekday())
        elif unit == "month":
            day = _month_span(date(today.year, today.month, 1))[1]
        else:
            day = date(today.year, 12, 31)
        return today, day
    if kind == "weekday":
        which, name = groups[0].lower(), groups[1].lower()
        ahead = (_WEEKDAYS[name] - today.weekday()) % 7
        if which == "next" and ahead == 0:
            ahead = 7
        day = today + timedelta(days=ahead)
        if which in ("by", "before"):
            return today, day
        return day, day
    if kind == "next_unit":
        which, unit = groups[0].lower(), groups[1].lower()
        if unit == "week":
            monday = today - timedelta(days=today.weekday())
            if which == "this":
                return today, monday + timedelta(days=6)
            return monday + timedelta(weeks=1), monday + timedelta(days=13)
        if unit == "weekend":
            saturday = today + timedelta(days=(5 - today.weekday()) % 7)
            if which == "next" and today.weekday() < 5:
                saturday += timedelta(weeks=1)
            return saturday, saturday + timedelta(days=1)
        if unit == "month":
            first = date(today.year, today.month, 1)
            if which == "this":
                return today, _month_span(first)[1]
            return _month_span(_add_months(first, 1))
        if which == "this":
            return today, date(today.year, 12, 31)
        return date(today.year + 1, 1, 1), date(today.year + 1, 12, 31)
    if kind == "in_month":
        month = _month(groups[0])
        year = today.year if month >= today.month else today.year + 1
        first, last = _month_span(date(year, month, 1))
        return max(first, today), last
    return None


def future_window(text: str, now) -> Optional[Tuple[date, date]]:
    """The first time ``text``, said on ``now``, points ahead to, as its
    first and last day: "tomorrow" said on 2024-03-01 gives 2024-03-02 for
    both, "by Friday" runs from the day it was said to that Friday, "next
    week" is the Monday to Sunday after. ``None`` when it names no future
    time, or ``now`` is not known."""
    today = as_date(now)
    if today is None or not text:
        return None
    found: List[Tuple[int, Tuple[date, date]]] = []
    taken: List[Tuple[int, int]] = []
    for kind, pattern in _FUTURE:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < b and a < end for a, b in taken):
                continue
            taken.append((start, end))
            span = _resolve(kind, match.groups(), today)
            if span and span[1] >= today:
                found.append((start, span))
    return min(found)[1] if found else None


@dataclass
class IntentionDraft:
    """An intention found in a message, before it is stored."""
    objective: str
    due: Optional[Tuple[date, date]]
    cue: str        # "reminder" | "commitment"


def detect_intentions(text: str, said_at=None) -> List[IntentionDraft]:
    """The intentions in a user message said at ``said_at``: each sentence
    with a reminder cue, and each sentence with a commitment cue and a future
    time. A question ("should I call her tomorrow?") is not a commitment."""
    out: List[IntentionDraft] = []
    for sentence in _SENTENCE.split((text or "").strip()):
        sentence = sentence.strip()
        if len(sentence.split()) < 3 or CANCEL_CUES.search(sentence):
            continue  # "no need to remind me to ..." takes one back
        due = future_window(sentence, said_at) if said_at is not None else None
        if REMINDER_CUES.search(sentence):
            out.append(IntentionDraft(sentence, due, "reminder"))
        elif (COMMITMENT_CUES.search(sentence) and due is not None
              and not sentence.rstrip().endswith("?")):
            out.append(IntentionDraft(sentence, due, "commitment"))
    return out


def detect_closure(text: str) -> Optional[Tuple[str, str]]:
    """``("done" | "cancelled", sentence)`` for the first sentence of a user
    message that says something was carried out or is no longer wanted;
    ``None`` otherwise. Which intention it closes is decided later by
    meaning (``IntentionLedger.maintain``)."""
    for sentence in _SENTENCE.split((text or "").strip()):
        sentence = sentence.strip()
        if CANCEL_CUES.search(sentence):
            return "cancelled", sentence
        if DONE_CUES.search(sentence):
            return "done", sentence
    return None


READINESS_SYSTEM = (
    "You decide which of a user's pending intentions an assistant should act on now. "
    "You get the current date, what is happening in the conversation, and numbered "
    "intentions, each with when it was said and when it is due.\n"
    "Write one line per intention and nothing else: the number, a colon, a few words "
    "of reasoning, then one verdict in capitals:\n"
    "ACT (remind or act now), WAIT (not yet), OBSERVE (relevant to what is happening, "
    "but no action is due), SATISFIED (the conversation shows it was done) or "
    "INVALIDATED (the conversation shows it is no longer wanted).\n"
    "When unsure, answer WAIT."
)
_VERDICT_LINE = re.compile(r"^\W*(\d+)\s*[:.)\-](.*)$", re.M)
_VERDICT_WORD = re.compile(r"\b(ACT|WAIT|OBSERVE|SATISFIED|INVALIDATED)\b")
VERDICTS = ("act", "wait", "observe", "satisfied", "invalidated")


def readiness_request(intentions: Sequence[Dict], context: Optional[str], today: str) -> str:
    lines = []
    for n, item in enumerate(intentions, 1):
        due = item.get("due_from") or "no date"
        if item.get("due_until") and item.get("due_until") != item.get("due_from"):
            due = f"{due} to {item['due_until']}"
        lines.append(f"{n}. said {item.get('said_on') or 'unknown'}, due {due}: {item['text']}")
    return (f"Current date: {today}\nConversation now: {context or '(nothing)'}\n\n"
            "Intentions:\n" + "\n".join(lines))


def parse_readiness(reply: Optional[str], count: int) -> Dict[int, str]:
    """Verdicts by position (0-based) from a reply to ``READINESS_SYSTEM``."""
    out: Dict[int, str] = {}
    for match in _VERDICT_LINE.finditer(reply or ""):
        k = int(match.group(1)) - 1
        word = _VERDICT_WORD.findall(match.group(2).upper())
        if 0 <= k < count and word:
            out[k] = word[-1].lower()
    return out


class LLMReadinessJudge:
    """Judges pending intentions as act / wait / observe / satisfied /
    invalidated with a chat model (``READINESS_SYSTEM``), one request per
    ``due()`` call that has candidates. Replies are cached with
    ``cache_dir``."""

    def __init__(self, model: str = "gpt-4o-mini", client=None,
                 base_url: Optional[str] = None, api_key: Optional[str] = None,
                 cache_dir: Optional[str] = None, **chat_kwargs):
        from ._chat import CachedChat

        self._chat = CachedChat(READINESS_SYSTEM, model=model, client=client,
                                base_url=base_url, api_key=api_key, cache_dir=cache_dir,
                                cache_name="readiness", max_tokens=400,
                                what="readiness check", **chat_kwargs)

    def forget_texts(self, texts) -> int:
        """Remove cached replies that carry ``texts`` (``Memory.forget``)."""
        return self._chat.forget_texts(texts)

    def judge(self, intentions: Sequence[Dict], context: Optional[str],
              today: str) -> Dict[int, str]:
        reply = self._chat.ask(readiness_request(intentions, context, today))
        return parse_readiness(reply, len(intentions))

    def flush_cache(self) -> None:
        self._chat.flush_cache()


def _filter(*parts: Dict) -> str:
    return json.dumps(parts[0] if len(parts) == 1 else {"op": "and", "filters": list(parts)})


def _eq(field: str, value) -> Dict:
    return {"op": "eq", "field": field, "value": value}


PENDING = (_eq("kind", "intention"), _eq("status", "pending"))


def _day(value) -> Optional[date]:
    return date.fromordinal(int(value)) if isinstance(value, int) else None


class IntentionLedger:
    """The intentions engine and its lifecycle. ``embed`` maps texts to
    document vectors (the Memory's embedder); every write is under the
    caller's lock."""

    def __init__(self, engine, embed: Callable[[List[str]], np.ndarray]):
        self.engine = engine
        self._embed = embed

    # -- formation ---------------------------------------------------------------
    def _next_ids(self, n: int) -> List[str]:
        seq = int(self.engine.next_insert_seq())
        ids = []
        while len(ids) < n:
            candidate = f"int_{seq}"
            seq += 1
            if not self.engine.contains_id(candidate):
                ids.append(candidate)
        return ids

    def record(self, user_id: Optional[str], drafts: Sequence[IntentionDraft],
               source_ids: Sequence[str], said_at) -> List[str]:
        """Store ``drafts`` as pending intentions of ``user_id``."""
        if not drafts:
            return []
        said = as_date(said_at)
        payloads = []
        seq = int(self.engine.next_insert_seq())
        for k, draft in enumerate(drafts):
            payload = {"kind": "intention", "status": "pending", "cue": draft.cue,
                       "user_id": user_id, "source_ids": list(source_ids),
                       "said_at": said_at if isinstance(said_at, (str, int, float)) else None,
                       "said_on": said.isoformat() if said else None,
                       "seq": seq + k, "checked": "no"}
            if draft.due is not None:
                first, last = draft.due
                payload.update(due_from=first.isoformat(), due_until=last.isoformat(),
                               due_from_day=first.toordinal(), due_until_day=last.toordinal())
            payloads.append(json.dumps(payload))
        return self._insert(user_id, [d.objective for d in drafts], payloads, "intention")

    def record_closure(self, user_id: Optional[str], kind: str, sentence: str,
                       source_ids: Sequence[str], said_at) -> List[str]:
        """Store a "done" / "cancelled" statement for ``maintain`` to match."""
        said = as_date(said_at)
        payload = {"kind": "signal", "status": "new", "closes_as": kind, "user_id": user_id,
                   "source_ids": list(source_ids),
                   "said_on": said.isoformat() if said else None,
                   "seq": int(self.engine.next_insert_seq())}
        return self._insert(user_id, [sentence], [json.dumps(payload)], "signal")

    def _insert(self, user_id, texts: List[str], payloads: List[str], role: str) -> List[str]:
        vectors = np.ascontiguousarray(self._embed(texts), dtype=np.float32)
        ids = self._next_ids(len(texts))
        self.engine.insert_batch(ids, texts, vectors, [1.0] * len(texts),
                                 [[] for _ in texts], payloads,
                                 [user_id] * len(texts) if user_id is not None else None,
                                 [role] * len(texts))
        return ids

    # -- reading -----------------------------------------------------------------
    def _items(self, ids: Sequence[str]) -> Dict[str, Dict]:
        out = {}
        for rec in self.engine.get_records(list(ids)):
            if rec is None:
                continue
            try:
                payload = json.loads(rec["payload"]) if rec["payload"] else {}
            except ValueError:
                payload = {}
            out[rec["id"]] = {"id": rec["id"], "text": rec["text"], "scope": rec["scope"],
                              **{k: v for k, v in payload.items()}}
        return out

    def pending(self, user_id: Optional[str]) -> List[Dict]:
        ids = self.engine.ids_where(_filter(*PENDING), user_id)
        items = self._items(ids)
        return [items[i] for i in ids if i in items]

    def due(self, user_id: Optional[str], now, context_vector=None,
            cue_min_cosine: float = CUE_MIN_COSINE, top_k: int = 8) -> List[Dict]:
        """Pending intentions whose time has come (``verdict="act"``, oldest
        due first, ``overdue`` once their last day has passed) and, given the
        vector of the current context, those it brings to mind
        (``verdict="observe"``, best match first). At most ``top_k``."""
        today = as_date(now)
        found: Dict[str, Dict] = {}
        if today is not None:
            time_due = self.engine.ids_where(_filter(*PENDING, {
                "op": "range", "field": "due_from_day", "high": today.toordinal()}), user_id)
            for item in self._items(time_due).values():
                item["verdict"] = "act"
                until = item.get("due_until_day")
                item["overdue"] = isinstance(until, int) and until < today.toordinal()
                found[item["id"]] = item
        if context_vector is not None:
            hits = self.engine.search_ann(np.asarray(context_vector, dtype=np.float32),
                                          max(2 * top_k, 16), scope=user_id,
                                          filter=_filter(*PENDING))
            strong = {mid: float(score) for mid, score in hits if score >= cue_min_cosine}
            for mid, item in self._items(list(strong)).items():
                if mid in found:
                    found[mid]["cue_score"] = strong[mid]
                    continue
                item.update(verdict="observe", overdue=False, cue_score=strong[mid])
                found[mid] = item
        ordered = sorted(found.values(), key=lambda i: (
            i["verdict"] != "act", i.get("due_from_day") or 0, -(i.get("cue_score") or 0.0)))
        return ordered[:max(0, int(top_k))]

    # -- lifecycle ---------------------------------------------------------------
    def set_status(self, updates: Sequence[Tuple[str, Dict]]) -> int:
        """Merge ``fields`` into each intention's payload: ``[(id, fields)]``,
        one batch of payload patches (the records keep their place)."""
        items = self._items([mid for mid, _ in updates])
        ids, payloads = [], []
        for mid, fields in updates:
            item = items.get(mid)
            if item is None:
                continue
            payload = {k: v for k, v in item.items() if k not in ("id", "text", "scope")}
            payload.update(fields)
            items[mid] = {**item, **fields}
            ids.append(mid)
            payloads.append(json.dumps(payload))
        return int(self.engine.set_payloads(ids, payloads)) if ids else 0

    def maintain(self, completion_min_cosine: float = COMPLETION_MIN_COSINE,
                 reschedule_min_cosine: float = RESCHEDULE_MIN_COSINE) -> Dict[str, int]:
        """One maintenance pass, all of it in one batch of payload patches:
        each new closure statement closes the nearest older pending intention
        of its user, and each new intention replaces an older pending one it
        restates. Returns counts of what changed."""
        updates: List[Tuple[str, Dict]] = []
        closed = replaced = 0
        taken = set()
        signals = self._items(self.engine.ids_where(
            _filter(_eq("kind", "signal"), _eq("status", "new"))))
        fresh = self._items(self.engine.ids_where(
            _filter(*PENDING, _eq("checked", "no"))))
        ids = list(signals) + list(fresh)
        vectors, live = self.engine.get_vectors(ids) if ids else (None, [])
        row = {mid: k for k, mid in enumerate(ids) if live[k]}

        def older_pending(mid: str, item: Dict, floor: float) -> Optional[str]:
            if mid not in row:
                return None
            hits = self.engine.search_ann(vectors[row[mid]], 8, scope=item.get("scope"),
                                          filter=_filter(*PENDING))
            for other, score in hits:
                if other == mid or other in taken or score < floor:
                    continue
                if self._items([other]).get(other, {}).get("seq", 0) < item.get("seq", 0):
                    return other
            return None

        for mid in sorted(signals, key=lambda m: signals[m].get("seq", 0)):
            item = signals[mid]
            target = older_pending(mid, item, completion_min_cosine)
            if target is not None:
                taken.add(target)
                status = "done" if item.get("closes_as") == "done" else "cancelled"
                updates.append((target, {"status": status, "closed_by": mid,
                                         "closed_on": item.get("said_on")}))
                closed += 1
            updates.append((mid, {"status": "applied" if target else "unmatched",
                                  "closes": target}))
        for mid in sorted(fresh, key=lambda m: fresh[m].get("seq", 0)):
            if mid in taken:
                continue
            target = older_pending(mid, fresh[mid], reschedule_min_cosine)
            if target is not None:
                taken.add(target)
                updates.append((target, {"status": "superseded", "superseded_by": mid}))
                replaced += 1
            updates.append((mid, {"checked": "yes"}))
        changed = self.set_status(_merge(updates)) if updates else 0
        return {"closed": closed, "replaced": replaced, "patched": changed}


def _merge(updates: Sequence[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
    """One patch per id, later fields over earlier ones, in first-seen order."""
    merged: Dict[str, Dict] = {}
    for mid, fields in updates:
        merged.setdefault(mid, {}).update(fields)
    return list(merged.items())
