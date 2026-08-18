"""OpenAI embedding backend (default embedder for ``tsm.Memory``).

Wraps the OpenAI embeddings API behind the ``Embedder`` protocol
(``encode()`` + ``dimension``). Vectors are cached to disk (keyed by
model+text) so repeated runs cost nothing. The API key is read from the
``OPENAI_API_KEY`` environment variable only — it is never handled or logged.
The ``openai`` package is imported lazily, so ``tsm`` imports fine without it.

text-embedding-3 vectors are already unit-normalized, so cosine == dot.
"""

import atexit
import logging
import os
import pickle
import time
from typing import List, Optional

import numpy as np

logger = logging.getLogger("tsm.embedders")

# Native output dimensions per model (used to size the engine's index).
_MODEL_DIM = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder:
    def __init__(self, model="text-embedding-3-small", dim=None, batch=256,
                 max_retries=6, request_timeout=30.0, cache_dir=None):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. The default OpenAIEmbedder needs "
                "it; pass a custom embedder to tsm.Memory to use another "
                "backend (see tsm.interfaces.Embedder)."
            )
        from openai import OpenAI

        self._client = OpenAI(timeout=request_timeout)
        self.model = model
        self._dim = dim or _MODEL_DIM.get(model, 1536)
        self.batch = batch
        self.max_retries = max_retries
        self.calls = 0
        self._cache = {}
        self._dirty = 0
        cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "tsm")
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

    # Embedder protocol -----------------------------------------------------------
    @property
    def dimension(self):
        return self._dim

    def get_sentence_embedding_dimension(self):
        """SentenceTransformer-compatible alias."""
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
                missing.append(t)
                seen.add(t)
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


class SentenceTransformerEmbedder:
    """Local open-source embedding backend using ``sentence-transformers`` models."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        get_dim = getattr(self.model, "get_embedding_dimension", getattr(self.model, "get_sentence_embedding_dimension", None))
        self._dim = get_dim() if get_dim else 384

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, texts):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        embs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embs = np.asarray(embs, dtype=np.float32)
        return embs[0] if single else embs
