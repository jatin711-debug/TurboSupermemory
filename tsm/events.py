"""Events: what a user reports doing, buying, attending or owning, one record each.

A question that counts or totals ("how many fish tanks do I have", "how much
did I spend on bike parts", "how many hours did I jog last week") needs every
instance of the thing, and they are spread over many conversations, each one
an aside in a long message about something else. A context of whole passages
holds a dozen or two of them; a reader then counts what it was given. On
LongMemEval-S the questions still answered wrong at 2,000 tokens were mostly
these, and neither a larger context nor a stronger reader helped (2026-11-05,
``benchmarks/PHASE_PROGRESS.md``).

``OpenAIEventExtractor`` lists the events a user message reports, each as

- ``what``: a short self-contained description naming the specific thing,
- ``kind``: what a question would count it by ("fish tank", "cuisine"),
- ``amount`` / ``unit``: a quantity the message gives (money, hours, items),
- ``when``: when it happened, as written ("last Saturday"); the date it
  points to is worked out at read time against the date it was said
  (``tsm.when``), so a reply is cached by the message alone,
- ``status``: ``done``, ``planned`` or ``has`` (owns or is doing it).

``Memory(events=...)`` stores each event as a short record in an engine of
its own (``EVENT_DIR`` inside the database), under the user's scope. The
engine's word index and concept graph are shared by all of an engine's
scopes, so events kept beside the memories changed every ordinary search
(17 of 20 contexts that had no events in them, 2026-11-05); apart, they
change nothing but the counting questions. ``recall()`` of a counting question (``tsm.ranking.is_counting_query``)
searches them as well and gives the best a share of the token budget
(``EVENT_SHARE``), each as one dated line (``event_note``). Twenty of those fit
where two passages would.

Nothing here touches the engine; only the extractor calls a model.
"""

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

from ._retry import call_with_retries
from .extractors import _cache_key, _prompt_id
from .ranking import is_first_person_query
from .when import annotate, said_on

logger = logging.getLogger("tsm.events")

# The role an event record is stored under, and the directory inside the
# database that holds the events' engine.
EVENT_ROLE = "event"
EVENT_DIR = "tsm_events"
# Share of a counting question's token budget that events may take.
EVENT_SHARE = 0.3
# How many events a counting question searches.
EVENT_POOL = 60
# How many messages before one are shown to resolve what it refers to, and
# how much of the end of each (an assistant reply runs to a thousand tokens;
# the part a reply to it refers to is mostly its end).
EVENT_CONTEXT = 2
CONTEXT_CHARS = 300
STATUSES = ("done", "planned", "has")

_SYS = (
    "You list the events a user reports in one message to an assistant, so that "
    "questions such as \"how many ... have I ...\", \"how much did I spend on ...\" or "
    "\"how many hours did I ...\" can later be answered by counting them.\n"
    "- An event is something the user (or their family, partner or pet, with the "
    "user involved) did, attended, bought, sold, made, cooked, received, gave, "
    "visited, read, watched, played, learned, started, finished, is working on or "
    "leading, owns, or plans to do.\n"
    "- Read the whole message. Most events are mentioned in passing, in a clause "
    "before or after a question (\"I recently finished a Tamiya Spitfire and need "
    "tips on weathering\" reports finishing the Spitfire; \"I'm working on a "
    "project analyzing customer data\" reports that project).\n"
    "- One entry per occasion and per named item: \"I adopted two cats, Luna and "
    "Max\" is two entries. Never list items the message does not name: \"watched "
    "all the main Star Wars films in a week and a half\" is one entry, with its "
    "amount 1.5 and unit \"weeks\".\n"
    "- \"what\": a short description that stands on its own, naming the specific "
    "thing (title, name, brand, place, dish), e.g. \"bought a 20-gallon fish tank "
    "for the guppies\".\n"
    "- \"kind\": what a question would count it by, one to three words, singular, "
    "e.g. \"fish tank\", \"cuisine\", \"concert\", \"book\", \"property viewing\", "
    "\"bike part\", \"jog\".\n"
    "- \"amount\": a number the message gives for it (money spent, time taken, "
    "miles, pages, items, people) or null; \"unit\": its unit (\"USD\", "
    "\"hours\", \"days\", \"miles\") or null.\n"
    "- \"when\": when it happened, exactly as written (\"last Saturday\", \"two "
    "weeks ago\", \"on March 3rd\", \"today\"), or null.\n"
    "- \"status\": \"done\" when it happened (bought, got, started, finished, "
    "attended), \"has\" when the user owns it or is in the middle of it now, "
    "\"planned\" when it has not happened yet.\n"
    "- Not events: what the user asks about or wants advice on, hypotheticals, "
    "general knowledge, and anything said only in \"Recent context\", which is "
    "there to resolve what the message refers to.\n"
    "Reply with a JSON object {\"events\": [{\"what\": ..., \"kind\": ..., "
    "\"amount\": ..., \"unit\": ..., \"when\": ..., \"status\": ...}]}. Use an "
    "empty list when the message reports none, as most messages that only ask "
    "something do."
)


def is_event(role: Optional[str]) -> bool:
    return role == EVENT_ROLE


def _clean(value) -> Optional[str]:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text if text and text.lower() not in ("null", "none", "n/a") else None


def _amount(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        text = str(value).replace(",", "").replace("$", "").strip()
        number = float(text)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def normalize(event) -> Optional[Dict]:
    """An event from a model's reply in the stored shape, or ``None`` when it
    has no description."""
    if not isinstance(event, dict):
        return None
    what = _clean(event.get("what"))
    if not what:
        return None
    status = (_clean(event.get("status")) or "done").lower()
    return {
        "what": what,
        "kind": _clean(event.get("kind")),
        "amount": _amount(event.get("amount")),
        "unit": _clean(event.get("unit")),
        "when": _clean(event.get("when")),
        "status": status if status in STATUSES else "done",
    }


def event_text(event: Dict) -> str:
    """What an event is stored and embedded as: its kind, then what happened."""
    kind = event.get("kind")
    return f"{kind}: {event['what']}" if kind else event["what"]


def event_note(event: Dict, timestamp=None) -> str:
    """How an event reads in a context, after its time tag:
    ``event: bought a 20-gallon fish tank (fish tank; 1 tank; last week [week of
    2023-04-24])``. A relative ``when`` is followed by the date it points to."""
    details = []
    if event.get("kind"):
        details.append(event["kind"])
    if event.get("amount") is not None:
        details.append(" ".join(str(x) for x in (event["amount"], event.get("unit")) if x))
    if event.get("when"):
        details.append(annotate(event["when"], said_on(timestamp)))
    status = event.get("status") or "done"
    head = "event" if status == "done" else f"event ({status})"
    tail = f" ({'; '.join(details)})" if details else ""
    return f"{head}: {event['what']}{tail}"


class OpenAIEventExtractor:
    """Lists the events in a user message with an OpenAI chat model (JSON
    mode), cached on disk by the message, the messages just before it and
    the prompt. ``extract_many`` asks for many messages at once."""

    def __init__(self, model: str = "gpt-4o-mini", max_retries: int = 6,
                 request_timeout: float = 30.0, cache_dir: Optional[str] = None,
                 max_tokens: int = 600, client=None, workers: int = 8):
        if client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. OpenAIEventExtractor needs it; pass "
                    "client=... for another OpenAI-compatible endpoint, or an event "
                    "extractor of your own (extract_events(message, context))."
                )
            from openai import OpenAI

            client = OpenAI(timeout=request_timeout)
        self._client = client
        self.model = model
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.workers = max(1, int(workers))
        self.calls = 0
        cdir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "tsm")
        os.makedirs(cdir, exist_ok=True)
        self._cache_path = os.path.join(
            cdir, f"events_{model.replace('/', '_').replace(':', '_')}.json")
        self._cache: Dict[str, List[Dict]] = {}
        if os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, encoding="utf-8") as fh:
                    self._cache = json.load(fh)
            except (OSError, json.JSONDecodeError):
                self._cache = {}
        self._dirty = 0
        self._lock = threading.RLock()

    @staticmethod
    def _recent(context: Optional[Sequence[str]]) -> List[str]:
        return [str(c)[-CONTEXT_CHARS:] for c in list(context or [])[-EVENT_CONTEXT:]]

    def cache_key(self, message: str, context: Optional[Sequence[str]] = None) -> str:
        return _cache_key(message, self._recent(context), prefix=_prompt_id(_SYS))

    def is_cached(self, message: str, context: Optional[Sequence[str]] = None) -> bool:
        return self.cache_key(message, context) in self._cache

    def _persist(self, force: bool = False) -> None:
        with self._lock:
            self._dirty += 1
            if not (force or self._dirty >= 200):
                return
            try:
                tmp = f"{self._cache_path}.{os.getpid()}.tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(self._cache, fh)
                os.replace(tmp, self._cache_path)
                self._dirty = 0
            except OSError as e:
                logger.warning("event cache write failed: %s", e)

    def _request(self, message: str, context: Sequence[str], max_tokens: int):
        ctx = ""
        if context:
            ctx = "Recent context:\n" + "\n".join(f"- {c}" for c in context) + "\n\n"
        user = f"{ctx}Message:\n\"{message}\""

        def request():
            with self._lock:
                self.calls += 1
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": _SYS},
                          {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            choice = resp.choices[0]
            return choice.message.content, getattr(choice, "finish_reason", None)

        return call_with_retries(request, "OpenAI event extraction", self.max_retries, logger)

    @staticmethod
    def _parse(raw: Optional[str]) -> Optional[List[Dict]]:
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        got = data.get("events", []) if isinstance(data, dict) else None
        if not isinstance(got, list):
            return None
        return [e for e in (normalize(item) for item in got) if e]

    def extract_events(self, message: str, context: Optional[Sequence[str]] = None
                       ) -> List[Dict]:
        """The events ``message`` reports. A message without a first-person
        word ("I", "my", "we" ...) reports none of the user's and is not
        asked (most such messages are requests: "write a poem about ...").
        A reply that could not be used is asked again with more room; after
        that the message gets no events (it is still stored as its facts and
        passages) and nothing is cached."""
        if not message or not message.strip() or not is_first_person_query(message):
            return []
        key = self.cache_key(message, context)
        cached = self._cache.get(key)
        if cached is not None:
            return [dict(e) for e in cached]
        recent = self._recent(context)
        for budget in (self.max_tokens, self.max_tokens * 4):
            raw, finish = self._request(message.strip(), recent, budget)
            parsed = self._parse(raw)
            if parsed is not None and finish != "length":
                with self._lock:
                    self._cache[key] = parsed
                    self._persist()
                return [dict(e) for e in parsed]
        logger.warning("event extraction reply was unusable for a %d-character message",
                       len(message))
        return []

    def extract_many(self, items: Sequence[Tuple[str, Sequence[str]]]) -> List[List[Dict]]:
        """``extract_events`` for each ``(message, context)``, ``workers`` at a
        time; the results are in the order given."""
        if self.workers == 1 or len(items) <= 1:
            return [self.extract_events(m, c) for m, c in items]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(lambda item: self.extract_events(*item), items))

    def flush_cache(self) -> None:
        with self._lock:
            if self._dirty:
                self._persist(force=True)
