"""One cached chat request, shared by the executive's model-backed parts.

The sufficiency judge (``tsm.monitor``), the readiness judge
(``tsm.intentions``) and the lesson distiller (``tsm.experience``) each ask a
chat model one short question and read one short reply. They share this
client: any OpenAI-compatible endpoint (OpenAI by default, a local server
through ``base_url``), the retry policy of ``tsm._retry`` (transient errors
back off, permanent ones and an empty account fail at once), temperature 0,
and an optional on-disk cache keyed by the instructions, the model and the
request, so the same question is only ever paid for once and a change of
instructions asks again instead of serving old replies.
"""

import hashlib
import json
import logging
import os
import threading
from typing import Dict, Optional

from ._retry import call_with_retries

logger = logging.getLogger("tsm.chat")


class CachedChat:
    """``ask(user) -> reply`` against one model with fixed instructions."""

    def __init__(self, system: str, model: str = "gpt-4o-mini", client=None,
                 base_url: Optional[str] = None, api_key: Optional[str] = None,
                 cache_dir: Optional[str] = None, cache_name: str = "chat",
                 max_retries: int = 4, request_timeout: float = 60.0,
                 max_tokens: int = 400, request_kwargs: Optional[dict] = None,
                 what: str = "chat request"):
        if client is None:
            if base_url is None and api_key is None and not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    f"OPENAI_API_KEY is not set. The {what} needs it, or pass "
                    "base_url=... (an OpenAI-compatible server) or client=...")
            from openai import OpenAI

            kwargs = {"timeout": request_timeout}
            if base_url is not None:
                kwargs["base_url"] = base_url
                kwargs["api_key"] = api_key or os.environ.get("OPENAI_API_KEY") or "unused"
            elif api_key is not None:
                kwargs["api_key"] = api_key
            client = OpenAI(**kwargs)
        self._client = client
        self.model = model
        self.system = system
        self.max_retries = max_retries
        self.max_tokens = int(max_tokens)
        self._request_kwargs = dict(request_kwargs or {})
        self._what = what
        self._lock = threading.Lock()
        self.calls = 0
        self._prompt_id = hashlib.sha256(system.encode("utf-8")).hexdigest()[:12]
        self._cache: Dict[str, str] = {}
        self._cache_path = None
        self._dirty = False
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            safe = model.replace("/", "_").replace(":", "_")
            self._cache_path = os.path.join(cache_dir, f"{cache_name}_{safe}.json")
            try:
                with open(self._cache_path, encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    self._cache = {k: v for k, v in loaded.items() if isinstance(v, str)}
            except (OSError, ValueError):
                self._cache = {}

    def _key(self, user: str) -> str:
        raw = "\x1f".join((self._prompt_id, self.model, user))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def ask(self, user: str) -> str:
        """The model's reply to ``user`` (cached when a cache is set). A
        request that fails after its retries raises."""
        key = self._key(user)
        with self._lock:
            if key in self._cache:
                return self._cache[key]

        def request():
            with self._lock:
                self.calls += 1
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.system},
                          {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=self.max_tokens,
                **self._request_kwargs,
            )
            return resp.choices[0].message.content or ""

        reply = call_with_retries(request, self._what, self.max_retries, logger)
        with self._lock:
            self._cache[key] = reply
            self._dirty = True
        return reply

    def forget_texts(self, texts) -> int:
        """Remove every cached reply that carries one of ``texts``
        (``tsm.forgetting``; the requests are only kept as hashes) and write
        the cache at once. Returns how many went."""
        from .forgetting import Forgotten

        forgotten = Forgotten(texts)
        if not forgotten:
            return 0
        with self._lock:
            gone = [key for key, reply in self._cache.items() if forgotten.carries(reply)]
            for key in gone:
                del self._cache[key]
            if gone:
                self._dirty = True
        if gone:
            self.flush_cache()
        return len(gone)

    def flush_cache(self) -> None:
        """Write the reply cache to disk (atomically) if it changed."""
        if not self._cache_path or not self._dirty:
            return
        with self._lock:
            snapshot = dict(self._cache)
            self._dirty = False
        tmp = self._cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(snapshot, fh)
        os.replace(tmp, self._cache_path)
