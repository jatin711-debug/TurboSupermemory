"""OpenAI-based atomic-fact extractor (default extractor for ``tsm.Memory``).

Uses JSON mode for reliable parsing and a persistent disk cache keyed by
message text (+ recent context) so the same message is never extracted (paid
for) twice. The API key is read from the ``OPENAI_API_KEY`` environment
variable only — it is never handled or logged. The ``openai`` package is
imported lazily, so ``tsm`` imports fine without it.

A message is never dropped because extraction went wrong: when the model's
reply is cut off or is not the expected JSON, the request is repeated once
with a larger reply budget, and if that fails too the message itself is
returned as a single fact. Only a well-formed reply is cached.
"""

import hashlib
import json
import logging
import os
from typing import List, Optional, Tuple

from ._retry import call_with_retries

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


def _cache_key(message: str, context: Optional[List[str]] = None) -> str:
    key = message.strip()
    if not context:
        return key
    recent = "\n".join(str(item) for item in context[-3:])
    digest = hashlib.sha256(f"{key}\0{recent}".encode("utf-8")).hexdigest()
    return f"context-v1:{digest}"


class OpenAIExtractor:
    def __init__(self, model: str = "gpt-4o-mini", max_retries: int = 6,
                 request_timeout: float = 30.0, cache_dir: str = None,
                 max_tokens: int = 400, client=None):
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

    def _persist(self, force=False):
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

    def _chat_json(self, message: str, context: Optional[List[str]],
                   max_tokens: int) -> Tuple[Optional[str], Optional[str]]:
        """One extraction request: ``(reply text, finish reason)``."""
        ctx = ""
        if context:
            ctx = "Recent context:\n" + "\n".join(f"- {c}" for c in context[-3:]) + "\n\n"
        user = f"{ctx}Message:\n\"{message}\""

        def request():
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

    # Extractor protocol ----------------------------------------------------------
    def extract_facts(self, message: str, context: Optional[List[str]] = None) -> List[str]:
        if not message or not message.strip():
            return []
        key = _cache_key(message, context)
        if key in self._cache:
            return self._cache[key]
        facts: Optional[List[str]] = None
        # A cut-off reply (finish reason "length") is not valid JSON, and a
        # refusal has no content at all. Either used to be read as "this
        # message contains no facts" and cached as such, so the message was
        # dropped now and on every later attempt. Ask again with room for a
        # longer reply before giving up on extraction.
        for budget in (self.max_tokens, self.max_tokens * 4):
            raw, finish = self._chat_json(message, context, budget)
            parsed = self._parse(raw)
            if parsed is not None and finish != "length":
                facts = parsed
                break
        if facts is None:
            logger.warning(
                "extraction reply was cut off or malformed for a %d-character message; "
                "storing the message itself as one fact", len(message))
            return [message.strip()]  # not cached: the next run may do better
        self._cache[key] = facts
        self._persist()
        return facts

    def flush_cache(self):
        """Write any unpersisted cache entries to disk."""
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


