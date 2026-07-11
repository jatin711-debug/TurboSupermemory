"""OpenAI-based atomic-fact extractor (W6).

Replaces the mock sentence-splitter with a real LLM extractor over the OpenAI
API — closing the biggest caveat in the eval record (mock extraction inflated
near-duplicate "facts" from assistant chatter). Same interface as
`OllamaExtractor` so it is a drop-in via the extractor factory.

Uses JSON mode for reliable parsing and an in-memory cache keyed by message text
so the identical corpus in the ON and OFF arms is only extracted once (halves
cost + time). The key is read from OPENAI_API_KEY in the environment and never
handled or logged.
"""

import json
import logging
import os
import time
from typing import List, Optional

logger = logging.getLogger("cognitive_eval.extraction.openai")

_SYS = (
    "You extract atomic facts from a single conversation message. An atomic fact "
    "is a self-contained statement that can stand alone. Break compound statements "
    "into multiple simple facts and preserve temporal cues (now, before, yesterday, "
    "no longer, ...). Extract only genuine facts the speaker asserts — ignore "
    "questions, pleasantries, and filler. Reply with a JSON object "
    '{"facts": ["...", "..."]}; use an empty list if there are none.'
)


class OpenAIExtractor:
    def __init__(self, model: str = "gpt-4o-mini", max_retries: int = 4, request_timeout: float = 30.0):
        from .._secrets import ensure_openai_key, key_file_hint
        if not ensure_openai_key():
            raise RuntimeError("No OpenAI key. " + key_file_hint())
        from openai import OpenAI

        self._client = OpenAI(timeout=request_timeout)
        self.model = model
        self.max_retries = max_retries
        self.calls = 0
        self._cache: dict = {}

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
                wait = 2.0 * (2 ** attempt)
                logger.warning("OpenAI extract failed (attempt %d/%d): %s; retry in %.0fs",
                               attempt + 1, self.max_retries, e, wait)
                time.sleep(wait)
        raise RuntimeError("OpenAI extraction failed after retries")

    def extract_facts(self, message: str, context: Optional[List[str]] = None) -> List[str]:
        if not message or not message.strip():
            return []
        key = message.strip()
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
        return facts

    def extract_facts_batch(self, messages, contexts=None):
        return [self.extract_facts(m, contexts[i] if contexts and i < len(contexts) else None)
                for i, m in enumerate(messages)]

    def health_check(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))
