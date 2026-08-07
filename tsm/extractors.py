"""OpenAI-based atomic-fact extractor (default extractor for ``tsm.Memory``).

Uses JSON mode for reliable parsing and a persistent disk cache keyed by
message text (+ recent context) so the same message is never extracted (paid
for) twice. The API key is read from the ``OPENAI_API_KEY`` environment
variable only — it is never handled or logged. The ``openai`` package is
imported lazily, so ``tsm`` imports fine without it.
"""

import hashlib
import json
import logging
import os
import time
from typing import List, Optional

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
                 request_timeout: float = 30.0, cache_dir: str = None):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. The default OpenAIExtractor needs "
                "it; pass a custom extractor to tsm.Memory to use another "
                "backend (see tsm.interfaces.Extractor)."
            )
        from openai import OpenAI

        self._client = OpenAI(timeout=request_timeout)
        self.model = model
        self.max_retries = max_retries
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
                with open(self._cache_path, "w", encoding="utf-8") as f:
                    json.dump(self._cache, f)
                self._dirty = 0
            except OSError as e:
                logger.warning("extract cache write failed: %s", e)

    def _chat_json(self, message: str, context: Optional[List[str]]) -> str:
        ctx = ""
        if context:
            ctx = "Recent context:\n" + "\n".join(f"- {c}" for c in context[-3:]) + "\n\n"
        user = f"{ctx}Message:\n\"{message}\""
        for attempt in range(self.max_retries):
            try:
                self.calls += 1
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": _SYS},
                              {"role": "user", "content": user}],
                    temperature=0.0,
                    max_tokens=400,
                    response_format={"type": "json_object"},
                )
                return resp.choices[0].message.content or "{}"
            except Exception as e:  # noqa: BLE001 — transient API errors: backoff + retry
                # 429s can be per-minute windows; back off long enough to outlast them.
                wait = min(5.0 * (2 ** attempt), 120.0)
                logger.warning("OpenAI extract failed (attempt %d/%d): %s; retry in %.0fs",
                               attempt + 1, self.max_retries, e, wait)
                time.sleep(wait)
        raise RuntimeError("OpenAI extraction failed after retries")

    # Extractor protocol ----------------------------------------------------------
    def extract_facts(self, message: str, context: Optional[List[str]] = None) -> List[str]:
        if not message or not message.strip():
            return []
        key = _cache_key(message, context)
        if key in self._cache:
            return self._cache[key]
        raw = self._chat_json(message, context)
        facts: List[str] = []
        try:
            data = json.loads(raw)
            got = data.get("facts", [])
            if isinstance(got, list):
                facts = [str(f).strip() for f in got if str(f).strip()]
        except json.JSONDecodeError:
            facts = []
        self._cache[key] = facts
        self._persist()
        return facts

    def flush_cache(self):
        """Write any unpersisted cache entries to disk."""
        self._persist(force=True)
