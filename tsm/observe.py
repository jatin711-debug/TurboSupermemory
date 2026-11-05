"""Observations: what a conversation leaves to remember, written down once.

A memory that keeps every message as said is complete but verbose: a request
for jewelry-cleaning tips and the assistant's eight-point answer carry one
clause worth keeping ("by the way, I just got a new silver necklace"), and a
context of such passages is mostly that kind of noise. A memory of atomic
facts per message is compact but lossy: the extractor found nothing in a
third of the user's messages. On LongMemEval-S most wrong answers had all
their evidence in a 2,000-token context of passages and the reader still
miscounted among the near-misses (2026-11-05).

Consolidation is the other way: after a conversation, write down what it
leaves to remember, densely and completely, and keep that as the long-term
record (Mastra's observational memory, Hindsight's narrative facts). Here an
``Observer`` reads one conversation (a day's messages) and returns
observations, one self-contained line each:

- everything the user reveals about their life, asides included, one
  occasion or item per line, numbers, names and titles kept exactly;
- what the assistant gave that may be asked about again, briefly;
- updates as updates ("now ..., previously ...");
- time expressions as written. The date a line points to is worked out at
  read time against the date the conversation happened (``tsm.when``), so
  an observation does not depend on that date and is cached by the
  conversation's text alone.

``observation_prompt`` and ``parse_observations`` call nothing;
``AnthropicObserver`` and ``OpenAIObserver`` ask a chat model, cache on disk,
and observe many conversations at once (``observe_many``).
"""

import hashlib
import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger("tsm.observe")

OBSERVE_SYSTEM = (
    "You keep the long-term memory of an assistant. You are given one conversation "
    "between a user and the assistant. Write down everything in it that could matter "
    "in a later conversation, as observations.\n"
    "- One observation per line, starting with \"- \". Each line must make sense on "
    "its own months later: name the people, places, things, titles, brands and numbers "
    "it is about, never \"it\", \"this\" or \"they\" without the noun. Call the user "
    "\"User\".\n"
    "- The user's own life comes first and is recorded completely: what they did, "
    "bought, got, made, cooked, attended, visited, read, watched, played, own, use, "
    "plan, prefer, dislike or feel; the people and pets in their life; amounts, prices, "
    "counts, durations, distances, scores and ages. Something the user mentions in "
    "passing while asking for something else (\"by the way, I just got a silver "
    "necklace\") matters as much as the main topic and must never be dropped.\n"
    "- One occasion or item per line: two purchases are two lines. Keep every number, "
    "unit, name and title exactly as written.\n"
    "- Keep time expressions exactly as written (\"last Saturday\", \"two weeks ago\", "
    "\"in March\", \"today\"); never convert them to dates.\n"
    "- When the user corrects or updates something, write the new state and what it "
    "replaced.\n"
    "- What the assistant gave that the user may come back to (specific "
    "recommendations, names, lists, figures, steps, a plan, a story or text it wrote) "
    "goes in one line each, briefly, starting with \"Assistant\".\n"
    "- A request on its own is worth a few words only when it shows an interest, a "
    "plan or a problem of the user's (\"User is preparing for a job interview at a "
    "bank\"). Generic advice the user did not take up is not worth recording.\n"
    "Write only the lines: no headings, no commentary. If nothing is worth "
    "remembering, write \"- (nothing)\"."
)

_LINE = re.compile(r"^\s*(?:[-*•]|\d{1,3}[.)])\s+(.*\S)\s*$")


def render_conversation(messages: Sequence[Dict]) -> str:
    """The conversation as the observer reads it: ``role: content`` blocks."""
    parts = []
    for msg in messages:
        content = (msg.get("content") or "").strip()
        if content:
            parts.append(f"{msg.get('role', 'user')}: {content}")
    return "\n\n".join(parts)


def observation_prompt(messages: Sequence[Dict]) -> List[Dict[str, str]]:
    """Chat messages asking for the observations of one conversation."""
    return [{"role": "system", "content": OBSERVE_SYSTEM},
            {"role": "user", "content": "Conversation:\n\n" + render_conversation(messages)}]


def parse_observations(reply: Optional[str]) -> List[str]:
    """The observation lines of a reply ("- ..." or numbered), without the
    marker; ``(nothing)`` and blank lines are dropped."""
    out = []
    for line in (reply or "").splitlines():
        match = _LINE.match(line)
        if not match:
            continue
        text = match.group(1).strip()
        if text and text.strip("().").lower() != "nothing":
            out.append(text)
    return out


def _key(prompt_text: str, model: str) -> str:
    prompt_id = hashlib.sha256(OBSERVE_SYSTEM.encode("utf-8")).hexdigest()[:8]
    digest = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    return f"p{prompt_id}:{model}:{digest}"


class _CachedObserver:
    """Disk cache, threads and the ``observe`` protocol; subclasses ask."""

    def __init__(self, model: str, cache_dir: Optional[str], workers: int, prefix: str):
        self.model = model
        self.workers = max(1, int(workers))
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.RLock()
        cdir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "tsm")
        os.makedirs(cdir, exist_ok=True)
        self._path = os.path.join(cdir, f"observe_{prefix}_{model.replace('/', '_')}.json")
        self._cache: Dict[str, List[str]] = {}
        if os.path.exists(self._path):
            try:
                with open(self._path, encoding="utf-8") as fh:
                    self._cache = json.load(fh)
            except (OSError, json.JSONDecodeError):
                self._cache = {}
        self._dirty = 0

    def _ask(self, system: str, user: str) -> Optional[str]:
        raise NotImplementedError

    def observe(self, messages: Sequence[Dict]) -> List[str]:
        """The observations of one conversation. A failed request raises; a
        conversation is never recorded as having nothing in it by mistake."""
        system, user = (m["content"] for m in observation_prompt(messages))
        if not render_conversation(messages):
            return []
        key = _key(user, self.model)
        cached = self._cache.get(key)
        if cached is not None:
            return list(cached)
        reply = self._ask(system, user)
        if reply is None:
            raise RuntimeError("observer reply was cut off or empty")
        found = parse_observations(reply)
        with self._lock:
            self._cache[key] = found
            self._dirty += 1
            if self._dirty >= 50:
                self._persist()
        return list(found)

    def observe_many(self, conversations: Sequence[Sequence[Dict]]) -> List[List[str]]:
        """``observe`` for each conversation, ``workers`` at a time, in order."""
        if self.workers == 1 or len(conversations) <= 1:
            return [self.observe(c) for c in conversations]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(self.observe, conversations))

    def _persist(self) -> None:
        with self._lock:
            try:
                tmp = f"{self._path}.{os.getpid()}.tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(self._cache, fh)
                os.replace(tmp, self._path)
                self._dirty = 0
            except OSError as e:
                logger.warning("observation cache write failed: %s", e)

    def flush_cache(self) -> None:
        with self._lock:
            if self._dirty:
                self._persist()


class AnthropicObserver(_CachedObserver):
    """Observes with a Claude model (default claude-haiku-5-5), thinking off.
    Needs the ``anthropic`` package and ``ANTHROPIC_API_KEY``."""

    def __init__(self, model: str = "claude-haiku-5-5", cache_dir: Optional[str] = None,
                 workers: int = 8, max_tokens: int = 4000, effort: str = "medium",
                 client=None, max_retries: int = 8):
        super().__init__(model, cache_dir, workers, "anthropic")
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=max_retries, timeout=180.0)
        self._client = client
        self.max_tokens = max_tokens
        self.effort = effort

    def _ask(self, system: str, user: str) -> Optional[str]:
        for budget in (self.max_tokens, self.max_tokens * 3):
            resp = self._client.messages.create(
                model=self.model, max_tokens=budget, system=system,
                messages=[{"role": "user", "content": user}],
                thinking={"type": "disabled"},
                extra_body={"output_config": {"effort": self.effort}})
            with self._lock:
                self.calls += 1
                self.input_tokens += resp.usage.input_tokens or 0
                self.output_tokens += resp.usage.output_tokens or 0
            if resp.stop_reason == "max_tokens":
                continue
            if resp.stop_reason == "refusal":
                return None
            return "".join(b.text for b in resp.content if b.type == "text")
        return None


class OpenAIObserver(_CachedObserver):
    """Observes with an OpenAI chat model (default gpt-4o-mini)."""

    def __init__(self, model: str = "gpt-4o-mini", cache_dir: Optional[str] = None,
                 workers: int = 8, max_tokens: int = 4000, client=None,
                 max_retries: int = 6):
        super().__init__(model, cache_dir, workers, "openai")
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=180.0)
        self._client = client
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def _ask(self, system: str, user: str) -> Optional[str]:
        from ._retry import call_with_retries

        for budget in (self.max_tokens, self.max_tokens * 3):
            def request(budget=budget):
                return self._client.chat.completions.create(
                    model=self.model, temperature=0.0, max_tokens=budget,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])

            resp = call_with_retries(request, "OpenAI observation", self.max_retries, logger)
            with self._lock:
                self.calls += 1
                if resp.usage:
                    self.input_tokens += resp.usage.prompt_tokens or 0
                    self.output_tokens += resp.usage.completion_tokens or 0
            choice = resp.choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                continue
            return choice.message.content or ""
        return None


# The role an observation is stored under, the directory inside the database
# that holds the observations' engine, and the most a conversation handed to
# the observer may hold (a longer stretch of one day is cut into pieces).
OBSERVATION_ROLE = "observation"
OBSERVATION_DIR = "tsm_observations"
OBSERVE_CHUNK_TOKENS = 12000


def conversations(messages: Sequence[Dict]) -> List[List[Dict]]:
    """Messages in the order said, cut into the conversations an observer
    reads: consecutive messages of one date (a message without a date joins
    the conversation before it), at most ``OBSERVE_CHUNK_TOKENS`` each."""
    from .budget import estimate_tokens
    from .when import said_on

    out: List[List[Dict]] = []
    day, size = object(), 0
    for msg in messages:
        when = said_on(msg.get("timestamp"))
        cost = estimate_tokens(msg.get("content") or "")
        new_day = when is not None and when != day
        if not out or new_day or size + cost > OBSERVE_CHUNK_TOKENS:
            out.append([])
            size = 0
            if when is not None:
                day = when
        out[-1].append(msg)
        size += cost
    return out


def observation_context(text: str, timestamp, time_tags=True) -> str:
    """How an observation reads in a context: the date of its conversation,
    then the line with each relative time followed by the date it points to."""
    from .when import said_on

    day = said_on(timestamp)
    shown = annotate_line(text, day)
    if time_tags and day is not None:
        return f"[{day.isoformat()}] {shown}"
    return shown


def annotate_line(text: str, day) -> str:
    from .when import annotate

    return annotate(text, day)
