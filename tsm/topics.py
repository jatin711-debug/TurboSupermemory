"""What a fact is about, so that an update worded unlike the fact it changes
can still be paired with it.

Belief revision shows a verifier each new fact next to its nearest older
facts by embedding. An update that shares no words or images with the fact
it replaces sits far from it: "I live in a small flat in Kreuzberg" and "I
moved to Lisbon last week" are 0.33 apart on OpenAI vectors, under the 0.45
floor, so the verifier never sees them. What the two have in common is not
their wording but what they give a value for: where the user lives.

A ``TopicKeyer`` names that: ``user.residence``, ``user.employer``,
``user_sister.residence``. Facts with the same key, of a property that holds
one value at a time, are offered to the verifier as candidates whatever
their cosine (``Memory(topics=...)``, at ``consolidate()``). The key only
proposes; the verifier still decides, so a wrong key costs a request, not a
fact.

``LLMTopicKeyer`` asks a chat model, many facts per request, and gives it the
keys it has already handed out so it reuses them. Answers are cached per fact
text, so a fact is keyed once.
"""

import hashlib
import json
import logging
import os
import re
import threading
from typing import Dict, List, Optional, Sequence, Tuple

from ._chat import CachedChat

logger = logging.getLogger("tsm.topics")

# A key, or None when the model gave no readable answer for that fact (asked
# again next time). ("", False) is "about no lasting property".
Topic = Optional[Tuple[str, bool]]

TOPIC_SYSTEM = """You label short statements from one person's memory with \
the property each one gives the CURRENT value of, so that a later statement \
that changes an earlier one can be found even when the two are worded \
nothing alike.

For each numbered statement answer one line, in order, starting with its \
number, in one of these forms:
N | one | subject.property | the property in a few words
N | many | subject.property | the property in a few words
N | past
N | none

For example, for "1. I moved to Lisbon last week. 2. We adopted a kitten. \
3. I was a smoker back in college. 4. I watched a film last night.":
1 | one | user.residence | where the user lives
2 | many | user.pets | the user's pets
3 | past
4 | none

Never write the value itself (not "Lisbon", not "kitten").

past: the statement says how things were during a time that is over, or \
when something happened long ago (back in college, as a teenager, in 2009, \
at a past event, from one year to another, at a given age). It does not \
give a current value, even when it names a property that has one (a past \
weight, a past job, a past diet).

none: a one-off event (I watched a film last night), a passing opinion, or \
a general fact about the world.

Otherwise the statement gives a current value of a property: anything that \
holds now and could later be different (where someone lives or works, an \
amount they pay or owe, a measurement, a provider or brand they use, a \
schedule, the date or place of something planned, a version in use, whether \
they do something).

The key is subject.property with exactly one dot. subject: "user" for the \
speaker (I, me, my, we, our); "user_" and the relation for someone or \
something of theirs (user_sister, user_dog, user_car); otherwise the name, \
lower case, words joined by underscores. property: a specific noun phrase, \
lower case, words joined by underscores (residence, employer, job_title, \
age, phone_number, bank, the name of a habit: user.cycling_habit). A habit \
and a statement that it stopped or started have the same property. Name the \
property, not the value: "I moved to Lisbon" and "I live in a flat in \
Kreuzberg" are both user.residence.

Reuse a key from KNOWN KEYS only when the statement gives a value for \
exactly that property. Two related things are still two properties (a \
salary and a bonus, a car and its insurance): make a new key rather than \
fit a statement under a broader one.

one: the subject has a single value of this property at a time (one home, \
one age, one job title, a habit is kept or dropped). many: several can hold \
at once (pets, hobbies, friends, trips, allergies, illnesses, degrees, \
languages, tools, accounts, things owned).

Answer every statement, one line each. Write nothing else."""

_LINE = re.compile(r"^\s*(\d+)\s*[|.:)\]-]\s*(.+?)\s*$")
_KEY = re.compile(r"^[a-z0-9_]+(?:\.[a-z0-9_]+)+$")


def normalize_key(raw: str) -> str:
    """Lower case, spaces and hyphens to underscores; "" when it is not a
    ``subject.property`` key."""
    key = re.sub(r"[\s\-]+", "_", (raw or "").strip().strip("`'\"").lower())
    return key if _KEY.match(key) else ""


def parse_topics(reply: str, count: int) -> Dict[int, Tuple[str, bool]]:
    """Read ``N | one|many | key`` / ``N | past`` / ``N | none`` lines; keyed
    by 0-based position (``past`` and ``none`` are both "no current value").
    A line that cannot be read is left out."""
    def read(body: str) -> Optional[Tuple[str, bool]]:
        # "one | user.residence", also the other way round, and with a value
        # written after the key ("one | user.residence | Lisbon").
        parts = [p.strip().lower().strip(".") for p in body.split("|")]
        if parts and parts[0] in ("none", "past"):
            return ("", False)
        cardinality = next((p for p in parts[:2] if p in ("one", "many")), None)
        key = next((normalize_key(p) for p in parts[:2] if normalize_key(p)), "")
        if key and cardinality:
            return (key, cardinality == "one")
        return None

    lines = [line.strip() for line in (reply or "").splitlines() if line.strip()]
    out: Dict[int, Tuple[str, bool]] = {}
    for line in lines:
        match = _LINE.match(line)
        if not match:
            continue
        index = int(match.group(1)) - 1
        if not 0 <= index < count or index in out:
            continue
        answer = read(match.group(2))
        if answer is not None:
            out[index] = answer
    if not out and len(lines) == count:
        # No numbers (a model that wrote "N | ..." as in a template): one
        # line per statement, in order, is still readable.
        for index, line in enumerate(lines):
            body = re.sub(r"^\s*(?:\d+|n)\s*[|.:)\]-]\s*", "", line, flags=re.IGNORECASE)
            answer = read(body)
            if answer is not None:
                out[index] = answer
    return out


class LLMTopicKeyer:
    """Keys facts with a chat model (any OpenAI-compatible endpoint).

    ``topics(texts, known)`` returns one ``(key, single_valued)`` per text,
    ``("", False)`` for a fact about no lasting property or one the reply
    had no readable line for. A request that fails raises (the facts are
    keyed on a later pass). ``batch_size`` facts go in one request.
    """

    def __init__(self, model: str = "gpt-4.1-mini", client=None,
                 base_url: Optional[str] = None, api_key: Optional[str] = None,
                 cache_dir: Optional[str] = None, batch_size: int = 10,
                 max_known: int = 80, per_record: int = 1, max_retries: int = 4,
                 request_timeout: float = 60.0, request_kwargs: Optional[dict] = None):
        self.batch_size = max(1, int(batch_size))
        # Older facts of the same key a new fact is paired with (latest first).
        self.per_record = max(1, int(per_record))
        self.max_known = int(max_known)
        # The request itself (retries, client); answers are cached here per
        # fact, not per request, so a fact is keyed once whatever batch it
        # came in.
        self._chat = CachedChat(TOPIC_SYSTEM, model=model, client=client, base_url=base_url,
                                api_key=api_key, cache_dir=None, max_retries=max_retries,
                                request_timeout=request_timeout,
                                max_tokens=40 * self.batch_size + 60,
                                request_kwargs=request_kwargs, what="topic keying")
        self.model = model
        self._prompt_id = hashlib.sha256(TOPIC_SYSTEM.encode("utf-8")).hexdigest()[:12]
        self._lock = threading.Lock()
        self._cache: Dict[str, str] = {}
        self._dirty = False
        self._cache_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            safe = model.replace("/", "_").replace(":", "_")
            self._cache_path = os.path.join(cache_dir, f"topics_{safe}.json")
            try:
                with open(self._cache_path, encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    self._cache = {k: v for k, v in loaded.items() if isinstance(v, str)}
            except (OSError, ValueError):
                self._cache = {}

    @property
    def calls(self) -> int:
        return self._chat.calls

    def _key(self, text: str) -> str:
        raw = "\x1f".join((self._prompt_id, self.model, text))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _decode(value: str) -> Tuple[str, bool]:
        key, _, one = value.partition("|")
        return key, one == "one"

    def topics(self, texts: Sequence[str], known: Sequence[str] = ()) -> List[Topic]:
        out: List[Topic] = [None] * len(texts)
        todo = []
        with self._lock:
            for i, text in enumerate(texts):
                cached = self._cache.get(self._key(text))
                if cached is not None:
                    out[i] = self._decode(cached)
                else:
                    todo.append(i)
        known_keys = list(dict.fromkeys(k for k in known if k))
        for start in range(0, len(todo), self.batch_size):
            chunk = todo[start:start + self.batch_size]
            listed = ", ".join(known_keys[-self.max_known:]) or "(none yet)"
            user = f"KNOWN KEYS: {listed}\n\n" + "\n".join(
                f"{n + 1}. {' '.join(texts[i].split())}" for n, i in enumerate(chunk))
            reply = self._chat.ask(user)
            answers = parse_topics(reply, len(chunk))
            # A fact the reply has no readable line for counts as about no
            # property: asking again would be paid for on every pass, and
            # the cosine candidates still cover it.
            for n in range(len(chunk)):
                answers.setdefault(n, ("", False))
            with self._lock:
                for n, (key, one) in answers.items():
                    i = chunk[n]
                    out[i] = (key, one)
                    self._cache[self._key(texts[i])] = f"{key}|{'one' if one else 'many'}"
                    self._dirty = True
            for key, _one in answers.values():
                if key and key not in known_keys:
                    known_keys.append(key)
        return out

    def forget_texts(self, texts) -> int:
        """Drop the cached keys of ``texts`` (``Memory.forget``)."""
        with self._lock:
            gone = [k for k in (self._key(t) for t in texts if t) if k in self._cache]
            for k in gone:
                del self._cache[k]
            if gone:
                self._dirty = True
        if gone:
            self.flush_cache()
        return len(gone)

    def flush_cache(self) -> None:
        if not self._cache_path or not self._dirty:
            return
        with self._lock:
            snapshot = dict(self._cache)
            self._dirty = False
        tmp = self._cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(snapshot, fh)
        os.replace(tmp, self._cache_path)


def topic_pairs(records: Sequence[Dict], per_record: int = 1) -> List[Tuple[str, str]]:
    """``(old_id, new_id)`` candidates from keyed records.

    ``records``: current (not superseded) records with ``id``, ``scope``,
    ``insert_seq`` and a parsed ``payload`` holding ``topic`` and
    ``topic_one``. Each record of a single-valued key is paired with the
    ``per_record`` latest older records of the same scope and key.
    """
    groups: Dict[Tuple, List[Dict]] = {}
    for rec in records:
        payload = rec.get("payload") or {}
        key = payload.get("topic")
        if key and payload.get("topic_one"):
            groups.setdefault((rec.get("scope"), key), []).append(rec)
    pairs: List[Tuple[str, str]] = []
    for group in groups.values():
        group.sort(key=lambda r: int(r["insert_seq"]))
        for i, rec in enumerate(group):
            for old in reversed(group[max(0, i - per_record):i]):
                pairs.append((old["id"], rec["id"]))
    return pairs
