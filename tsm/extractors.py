"""OpenAI-based atomic-fact extractor (default extractor for ``tsm.Memory``).

Uses JSON mode for reliable parsing and a persistent disk cache keyed by
message text (+ recent context) so the same message is never extracted (paid
for) twice. The API key is read from the ``OPENAI_API_KEY`` environment
variable only — it is never handled or logged. The ``openai`` package is
imported lazily, so ``tsm`` imports fine without it.

A message is never dropped because extraction went wrong: when the model's
reply is cut off or is not the expected JSON, the request is repeated once
with a larger reply budget, and if that fails too the message itself is
stored.

A message the model finds nothing in is another matter. People state things
about themselves while asking for something else ("any collar for a Golden
Retriever like Max?"), and a model asked for the facts a speaker asserts
returned an empty list for a third of the user messages when this was
measured. Two steps recover such a message, and both are off by default.
``second_look`` asks about it again with a second prompt, which asks what
the speaker reveals; ``keep_unextracted`` stores its statements as they were
said when the model still finds nothing. They are off because they did not
pay when judged (``benchmarks/PHASE_PROGRESS.md``, 2026-10-12): with a
600-token context and the ``tsm.reading`` prompt the recovered memories
changed 11 answers of 115, 6 for the better and 5 for the worse, and with a
150-token context they scored lower, because they compete with the other
facts for a few slots. Turn them on when recall has room and nothing a user
said may go unstored. The first prompt stays the first either way: asking
every message with the second one cost judged answers.

Only a well-formed reply is cached, under a key that includes the prompt, so
a reply to one prompt is never served for another.
"""

import hashlib
import json
import logging
import os
import re
import threading
from typing import List, Optional, Tuple

from ._retry import call_with_retries
from .budget import estimate_tokens

logger = logging.getLogger("tsm.extractors")

_SYS = (
    "You extract atomic facts from a single conversation message. An atomic fact "
    "is a self-contained statement that can stand alone. Break compound statements "
    "into multiple simple facts and preserve temporal cues (now, before, yesterday, "
    "no longer, ...). Extract only genuine facts the speaker asserts — ignore "
    "questions, pleasantries, and filler. Reply with a JSON object "
    '{"facts": ["...", "..."]}; every array element must be a string, never an '
    'object. Use an empty list if there are none.'
)

# Asked only about a message the prompt above found nothing in.
_SYS_SECOND = (
    "You extract atomic facts from one message of a conversation, so that they can "
    "be recalled later.\n"
    "- A fact is a short statement that can be understood without the message: name "
    "the people and things it is about instead of writing \"it\" or \"they\", call "
    "the person who wrote the message \"the speaker\", and break a compound "
    "statement into separate facts.\n"
    "- Extract what the speaker says or reveals about themselves and their world, "
    "even in passing: what they own, did, plan or prefer, people and pets, places, "
    "amounts, scores, durations, dates. A message that is mostly a question or a "
    "request usually reveals such facts too (\"any collar for a Golden Retriever "
    "like Max?\" reveals that the speaker has a Golden Retriever named Max), and an "
    "aside (\"by the way\", \"I just ...\") matters as much as the main point.\n"
    "- The question or request itself is not a fact, and neither are pleasantries: "
    "do not write facts that only say what the speaker is asking, wondering or "
    "looking for help with. A lasting preference, goal or plan is a fact.\n"
    "- If the message itself gives information (an explanation, a list, steps, "
    "recommendations), extract the specific items with their names and values.\n"
    "- Keep numbers, units, names, titles and dates exactly as written, and keep "
    "temporal cues (today, yesterday, last week, no longer, ...).\n"
    "- Extract only from the text under \"Message\". \"Recent context\" is there to "
    "resolve what the message refers to; never extract facts from it.\n"
    "Reply with a JSON object {\"facts\": [\"...\", \"...\"]}; every array element "
    "must be a string, never an object. Use an empty list only when the message "
    "reveals nothing."
)


def _prompt_id(prompt: str) -> str:
    """Part of every OpenAIExtractor cache key: a reply to one prompt is never
    served for another, so changing a prompt re-extracts."""
    return "p" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8] + ":"


# A message stored as it was said is cut into pieces of about this many
# tokens, so that one piece fits a small recall budget.
PIECE_TOKENS = 80
# A message with fewer words than this that yields no fact is left at that
# ("ok", "thanks!"): no second prompt, nothing stored.
MIN_WORDS_TO_KEEP = 3

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_LIST_NUMBER = re.compile(r"^\d{1,3}\.$")


def _sentences(line: str) -> List[str]:
    """A line cut at sentence ends. A list number stays with its item:
    "2. Which city?" is one sentence, not "2." and "Which city?", so the
    number is neither stored on its own nor cut off from what it numbers."""
    out: List[str] = []
    number = ""
    for part in _SENTENCE_END.split(line.strip()):
        part = part.strip()
        if not part:
            continue
        if number:
            part, number = f"{number} {part}", ""
        if _LIST_NUMBER.match(part):
            number = part
        else:
            out.append(part)
    if number:
        out.append(number)
    return out


def statements(message: str, max_tokens: int = PIECE_TOKENS) -> List[str]:
    """What a message says, without what it asks: its sentences that are not
    questions, as ``message_pieces``. Empty when fewer than
    ``MIN_WORDS_TO_KEEP`` words are left (the message only asked)."""
    kept = []
    for line in message.strip().splitlines():
        said = [s for s in _sentences(line) if not s.endswith("?")]
        if said:
            kept.append(" ".join(said))
    text = "\n".join(kept)
    if len(text.split()) < MIN_WORDS_TO_KEEP:
        return []
    return message_pieces(text, max_tokens)


def message_pieces(message: str, max_tokens: int = PIECE_TOKENS) -> List[str]:
    """A message as the memories it is stored as when no facts were extracted.

    The message itself when it fits ``max_tokens``; otherwise consecutive
    pieces of at most that size, cut at line ends, then at sentence ends,
    and inside a sentence only when one sentence is longer than a piece.
    Nothing is left out and nothing is reworded.
    """
    text = message.strip()
    if not text:
        return []
    if estimate_tokens(text) <= max_tokens:
        return [text]
    units: List[Tuple[str, str]] = []  # (text, what joins it to the unit before)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        joiner = "\n"
        for sentence in ([line] if estimate_tokens(line) <= max_tokens else _sentences(line)):
            if estimate_tokens(sentence) <= max_tokens:
                units.append((sentence, joiner))
            else:
                words, start = sentence.split(), 0
                while start < len(words):
                    stop = start + 1
                    while (stop < len(words)
                           and estimate_tokens(" ".join(words[start:stop + 1])) <= max_tokens):
                        stop += 1
                    units.append((" ".join(words[start:stop]), joiner))
                    start, joiner = stop, " "
            joiner = " "
    pieces: List[str] = []
    current = ""
    for unit, joiner in units:
        joined = f"{current}{joiner}{unit}" if current else unit
        if current and estimate_tokens(joined) > max_tokens:
            pieces.append(current)
            current = unit
        else:
            current = joined
    if current:
        pieces.append(current)
    return pieces


def _cache_key(message: str, context: Optional[List[str]] = None, prefix: str = "") -> str:
    key = message.strip()
    if context:
        recent = "\n".join(str(item) for item in context[-3:])
        digest = hashlib.sha256(f"{key}\0{recent}".encode("utf-8")).hexdigest()
        key = f"context-v1:{digest}"
    return f"{prefix}{key}"


class OpenAIExtractor:
    def __init__(self, model: str = "gpt-4o-mini", max_retries: int = 6,
                 request_timeout: float = 30.0, cache_dir: str = None,
                 max_tokens: int = 400, client=None, second_look: bool = False,
                 keep_unextracted: bool = False):
        """
        Args:
            second_look: when the first prompt finds no fact in a message of
                ``MIN_WORDS_TO_KEEP`` words or more, ask once more with the
                prompt for what the speaker reveals (one more request for
                such a message, about a quarter of all messages).
            keep_unextracted: when the model finds nothing, store what the
                message states as it was said (``statements``) instead of
                nothing. With both False, the default, only the first
                prompt's facts are stored.
        """
        if client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. The default OpenAIExtractor needs "
                    "it; pass client=... for another OpenAI-compatible endpoint, or "
                    "a custom extractor to tsm.Memory to use another backend "
                    "(see tsm.interfaces.Extractor)."
                )
            from openai import OpenAI

            client = OpenAI(timeout=request_timeout)
        self._client = client
        self.model = model
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.second_look = second_look
        self.keep_unextracted = keep_unextracted
        self.calls = 0
        cdir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "tsm")
        os.makedirs(cdir, exist_ok=True)
        self._cache_path = os.path.join(
            cdir, f"extract_{model.replace('/', '_').replace(':', '_')}.json")
        self._cache: dict = {}
        if os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info("Loaded %d cached extractions from %s",
                            len(self._cache), os.path.basename(self._cache_path))
            except (OSError, json.JSONDecodeError):
                self._cache = {}
        self._dirty = 0
        # The cache is shared by every thread that extracts through this
        # object (Memory.add may be called from several).
        self._lock = threading.RLock()

    def cache_key(self, message: str, context: Optional[List[str]] = None,
                  second: bool = False) -> str:
        """The key the reply about ``message`` is cached under, given the
        messages before it: the first prompt's reply, or the second's."""
        return _cache_key(message, context, prefix=_prompt_id(_SYS_SECOND if second else _SYS))

    def is_cached(self, message: str, context: Optional[List[str]] = None) -> bool:
        """True when ``extract_facts`` would answer without a request."""
        first = self._cache.get(self.cache_key(message, context))
        if first is None:
            return False
        if first or not self.second_look or len(message.split()) < MIN_WORDS_TO_KEEP:
            return True
        return self.cache_key(message, context, second=True) in self._cache

    def _persist(self, force=False):
        with self._lock:
            self._dirty += 1
            if force or self._dirty >= 200:
                try:
                    # Write beside the cache and rename: a crash mid-write must
                    # not leave a half-written file that resets the whole cache.
                    tmp = self._cache_path + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(self._cache, f)
                    os.replace(tmp, self._cache_path)
                    self._dirty = 0
                except OSError as e:
                    logger.warning("extract cache write failed: %s", e)

    def _chat_json(self, message: str, context: Optional[List[str]], max_tokens: int,
                   system: str = _SYS) -> Tuple[Optional[str], Optional[str]]:
        """One extraction request: ``(reply text, finish reason)``."""
        ctx = ""
        if context:
            ctx = "Recent context:\n" + "\n".join(f"- {c}" for c in context[-3:]) + "\n\n"
        user = f"{ctx}Message:\n\"{message}\""

        def request():
            with self._lock:
                self.calls += 1
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            choice = resp.choices[0]
            return choice.message.content, getattr(choice, "finish_reason", None)

        return call_with_retries(request, "OpenAI extraction", self.max_retries, logger)

    @staticmethod
    def _parse(raw: Optional[str]) -> Optional[List[str]]:
        """Facts from a reply, or ``None`` when the reply is not usable (no
        content, not JSON, or not the ``{"facts": [...]}`` shape)."""
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        got = data.get("facts", [])
        if not isinstance(got, list):
            return None
        return [str(f).strip() for f in got if str(f).strip()]

    def _ask(self, message: str, context: Optional[List[str]], second: bool
             ) -> Optional[List[str]]:
        """The facts one prompt finds in ``message``, from the cache or the
        model; ``None`` when the model's reply could not be used."""
        key = self.cache_key(message, context, second=second)
        facts: Optional[List[str]] = self._cache.get(key)
        if facts is not None:
            return facts
        # A cut-off reply (finish reason "length") is not valid JSON, and a
        # refusal has no content at all. Either used to be read as "this
        # message contains no facts" and cached as such, so the message was
        # dropped now and on every later attempt. Ask again with room for a
        # longer reply before giving up on extraction.
        for budget in (self.max_tokens, self.max_tokens * 4):
            raw, finish = self._chat_json(message, context, budget,
                                          _SYS_SECOND if second else _SYS)
            parsed = self._parse(raw)
            if parsed is not None and finish != "length":
                with self._lock:
                    self._cache[key] = parsed
                    self._persist()
                return parsed
        return None  # not cached: the next run may do better

    # Extractor protocol ----------------------------------------------------------
    def extract_facts(self, message: str, context: Optional[List[str]] = None) -> List[str]:
        if not message or not message.strip():
            return []
        facts = self._ask(message, context, second=False)
        if facts is None:
            logger.warning(
                "extraction reply was cut off or malformed for a %d-character message; "
                "storing the message itself", len(message))
            return message_pieces(message)
        if facts or len(message.split()) < MIN_WORDS_TO_KEEP:
            return list(facts)
        # The model found nothing to extract. That is right for "thanks!" and
        # wrong for a statement made while asking for something else, which
        # reads as "just a question" to an extractor. The two opt-in steps
        # below recover it; the empty reply stays cached, so the second look
        # costs at most one more request per such message.
        if self.second_look:
            revealed = self._ask(message, context, second=True)
            if revealed:
                return list(revealed)
        return statements(message) if self.keep_unextracted else []

    def flush_cache(self):
        """Write any unpersisted cache entries to disk."""
        with self._lock:
            if self._dirty:
                self._persist(force=True)


class GlinerExtractor:
    """Extracts structured atomic facts & entities using local GLiNER models.

    Supports Fastino's GLiNER 2.5 / GLiNER multi-task architecture running
    100% locally on CPU or CUDA without any external LLM API dependencies ($0.00 cost).
    """

    def __init__(
        self,
        model_name: str = "fastino/gliner2.5-multi-v1",
        schema: Optional[List[str]] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.schema = schema or [
            "user preference",
            "key fact",
            "technical specification",
            "person",
            "organization",
            "date and time",
            "location",
            "action",
            "constraint",
        ]
        self._device = device
        self._model = None
        cdir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "tsm")
        os.makedirs(cdir, exist_ok=True)
        self._cache_path = os.path.join(
            cdir, f"extract_gliner_{model_name.replace('/', '_').replace(':', '_')}.json"
        )
        self._cache: dict = {}
        if os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, encoding="utf-8") as f:
                    self._cache = json.load(f)
            except (OSError, json.JSONDecodeError):
                self._cache = {}
        self._dirty = 0

    def _load(self):
        if self._model is not None:
            return
        import torch

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading GLiNER model %s on %s", self.model_name, device)
        try:
            from gliner2 import GLiNER2
            self._model = GLiNER2.from_pretrained(self.model_name)
            if device == "cuda" and hasattr(self._model, "to"):
                self._model = self._model.to("cuda")
        except Exception:
            try:
                from gliner import GLiNER
                self._model = GLiNER.from_pretrained(self.model_name)
                if device == "cuda" and hasattr(self._model, "to"):
                    self._model = self._model.to("cuda")
            except Exception as e:
                logger.warning("Could not load GLiNER model %s: %s; falling back to urchade/gliner_multi-v2.1", self.model_name, e)
                from gliner import GLiNER
                self._model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
                if device == "cuda" and hasattr(self._model, "to"):
                    self._model = self._model.to("cuda")

    def extract_facts(self, message: str, context: Optional[List[str]] = None) -> List[str]:
        if not message or not message.strip():
            return []
        key = _cache_key(message, context)
        if key in self._cache:
            return self._cache[key]
        self._load()

        facts: List[str] = []
        try:
            if hasattr(self._model, "predict_entities"):
                extracted = self._model.predict_entities(message, self.schema)
            else:
                extracted = self._model.predict_entities(message, self.schema, threshold=0.3)
            
            if extracted:
                entity_summary = ", ".join(f"{e.get('label')}: {e.get('text')}" for e in extracted if 'label' in e and 'text' in e)
                facts.append(f"{message.strip()} [{entity_summary}]")
            else:
                facts.append(message.strip())
        except Exception as e:
            logger.warning("GLiNER extraction failed: %s", e)
            facts = [message.strip()]

        self._cache[key] = facts
        self._dirty += 1
        if self._dirty >= 100:
            self._persist()
        return facts

    def _persist(self, force=False):
        try:
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f)
            self._dirty = 0
        except OSError as e:
            logger.warning("GLiNER extract cache write failed: %s", e)

    def flush_cache(self):
        self._persist(force=True)


class PassthroughExtractor:
    """Zero-overhead extractor that treats the raw message as an atomic fact."""

    def extract_facts(self, message: str, context: Optional[List[str]] = None) -> List[str]:
        return [message.strip()] if message and message.strip() else []

    def flush_cache(self):
        pass


