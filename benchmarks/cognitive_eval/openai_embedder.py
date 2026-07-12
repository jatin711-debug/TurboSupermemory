"""OpenAI embedding backend for the TSM adapter (levels the A4 field).

The head-to-head (A4) let Mem0 use OpenAI `text-embedding-3-small` (1536-d) while
naive/TSM used a local 384-d MiniLM — an asymmetry that FAVORED Mem0. This wraps the
OpenAI embeddings API behind the SentenceTransformer-ish interface the adapter needs
(`encode()` + `get_sentence_embedding_dimension()`), so naive/TSM can run on the exact
same embeddings Mem0 uses. Vectors are cached to disk (keyed by model+text) so repeated
runs cost nothing; the key is read from the environment only. text-embedding-3 vectors
are already unit-normalized, so cosine == dot.
"""

import atexit
import logging
import os
import pickle
import time

import numpy as np

logger = logging.getLogger("cognitive_eval.openai_embedder")

# Native output dimensions per model (used to size the engine's index).
_MODEL_DIM = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder:
    def __init__(self, model="text-embedding-3-small", dim=None, batch=256,
                 max_retries=6, request_timeout=30.0, cache_dir=None):
        from ._secrets import ensure_openai_key, key_file_hint
        if not ensure_openai_key():
            raise RuntimeError("No OpenAI key. " + key_file_hint())
        from openai import OpenAI
        self._client = OpenAI(timeout=request_timeout)
        self.model = model
        self._dim = dim or _MODEL_DIM.get(model, 1536)
        self.batch = batch
        self.max_retries = max_retries
        self.calls = 0
        self._cache = {}
        self._dirty = 0
        cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "embedding", "_cache")
        os.makedirs(cache_dir, exist_ok=True)
        self._cache_path = os.path.join(cache_dir, f"emb_{model}.pkl")
        if os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, "rb") as f:
                    self._cache = pickle.load(f)
                logger.info("Loaded %d cached embeddings from %s",
                            len(self._cache), os.path.basename(self._cache_path))
            except Exception as e:  # noqa: BLE001
                logger.warning("embed cache load failed: %s", e)
                self._cache = {}
        atexit.register(self.flush)

    # SentenceTransformer-compatible surface -------------------------------------
    def get_sentence_embedding_dimension(self):
        return self._dim

    def encode(self, texts, **_kwargs):
        """Embed a single string (-> 1-D vec) or a list (-> 2-D array). Cache-backed;
        only uncached, de-duplicated texts hit the API."""
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        # OpenAI rejects empty input; map blanks to a single space (stable key).
        norm = [t if (t and t.strip()) else " " for t in items]

        missing, seen = [], set()
        for t in norm:
            if t not in self._cache and t not in seen:
                missing.append(t); seen.add(t)
        for i in range(0, len(missing), self.batch):
            chunk = missing[i:i + self.batch]
            for t, v in zip(chunk, self._embed_batch(chunk)):
                self._cache[t] = v
                self._dirty += 1
        if self._dirty >= 300:
            self.flush()

        out = np.vstack([self._cache[t] for t in norm]).astype(np.float32)
        return out[0] if single else out

    # internals ------------------------------------------------------------------
    def _embed_batch(self, chunk):
        for attempt in range(self.max_retries):
            try:
                self.calls += 1
                r = self._client.embeddings.create(model=self.model, input=chunk)
                return [np.asarray(d.embedding, dtype=np.float32) for d in r.data]
            except Exception as e:  # noqa: BLE001
                wait = min(5.0 * (2 ** attempt), 120.0)
                logger.warning("embed failed (attempt %d/%d): %s; retry %.0fs",
                               attempt + 1, self.max_retries, e, wait)
                time.sleep(wait)
        raise RuntimeError("embedding failed after retries")

    def flush(self):
        if not self._dirty:
            return
        try:
            tmp = self._cache_path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(self._cache, f)
            os.replace(tmp, self._cache_path)
            self._dirty = 0
        except Exception as e:  # noqa: BLE001
            logger.warning("embed cache flush failed: %s", e)
